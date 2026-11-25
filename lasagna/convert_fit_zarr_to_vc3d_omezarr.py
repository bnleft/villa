from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
import multiprocessing
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from vc_reader.reader import ZarrWriter, create_group, open_zarr

try:
	from omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		print_progress,
		shape_div2,
	)
except ImportError:
	from lasagna.omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		print_progress,
		shape_div2,
	)


def _ceil_div(a: int, b: int) -> int:
	return (int(a) + int(b) - 1) // int(b)


def _iter_bounded(pool: ThreadPoolExecutor, worker, work, *, max_in_flight: int):
	pending: deque[Future] = deque()
	it = iter(work)
	limit = max(1, int(max_in_flight))
	try:
		for _ in range(limit):
			try:
				item = next(it)
			except StopIteration:
				break
			pending.append(pool.submit(worker, item))
		while pending:
			res = pending.popleft().result()
			try:
				nxt = next(it)
			except StopIteration:
				nxt = None
			if nxt is not None:
				pending.append(pool.submit(worker, nxt))
			yield res
	finally:
		for fut in pending:
			fut.cancel()


def _run_with_progress(work, worker, *, workers: int, prefix: str) -> None:
	total = len(work)
	if total == 0:
		return
	t0 = time.time()
	print_progress(prefix=prefix, done=0, total=total, t0=t0)
	pool_size = min(max(1, int(workers)), total)
	with ThreadPoolExecutor(max_workers=pool_size) as pool:
		with closing(_iter_bounded(pool, worker, work, max_in_flight=pool_size * 2)) as results:
			for done, _ in enumerate(results, 1):
				print_progress(prefix=prefix, done=done, total=total, t0=t0)
	print("", flush=True)


def _channels_from_src(*, c_in: int, params: dict) -> list[str]:
	ch = params.get("channels", None)
	if isinstance(ch, list) and all(isinstance(v, str) for v in ch) and len(ch) == int(c_in):
		return [str(v) for v in ch]
	return [f"ch{i}" for i in range(int(c_in))]


def _first_filled_level_from_downscale(scaledown: int) -> int:
	d = max(1, int(scaledown))
	lv = 0
	while d > 1:
		d //= 2
		lv += 1
	return lv


def _upsample_level(*, src: ZarrWriter, dst: ZarrWriter, chunk: int, workers: int, prefix: str) -> None:
	"""Fill a level from the one above by nearest-neighbor 2x upsampling over stored source chunks."""
	scz, scy, scx = (int(v) for v in src.chunks)
	starts = [
		(iz * scz, iy * scy, ix * scx)
		for iz, iy, ix in sorted(src.stored_chunks())
	]
	if not starts:
		return

	def upsample(start):
		z0, y0, x0 = start
		z1 = min(src.shape[0], z0 + scz)
		y1 = min(src.shape[1], y0 + scy)
		x1 = min(src.shape[2], x0 + scx)
		slab = src[z0:z1, y0:y1, x0:x1]
		up = slab.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
		dz0, dy0, dx0 = z0 * 2, y0 * 2, x0 * 2
		uz = min(up.shape[0], dst.shape[0] - dz0)
		uy = min(up.shape[1], dst.shape[1] - dy0)
		ux = min(up.shape[2], dst.shape[2] - dx0)
		if uz > 0 and uy > 0 and ux > 0:
			dst[dz0:dz0 + uz, dy0:dy0 + uy, dx0:dx0 + ux] = up[:uz, :uy, :ux]

	_run_with_progress(starts, upsample, workers=workers, prefix=prefix)


def _aligned_crop_tiles(
	zs0: int, zs1: int, ys0: int, ys1: int, xs0: int, xs1: int,
	chunks_zyx: tuple[int, int, int],
) -> list[tuple[int, int, int, int, int, int]]:
	"""Partition [zs0,zs1)x[ys0,ys1)x[xs0,xs1) along destination chunk boundaries in bounded tiles."""
	cz, cy, cx = (max(1, int(v)) for v in chunks_zyx)
	step_z = cz
	step_y = cy * max(1, 256 // cy)
	step_x = cx * max(1, 256 // cx)

	def _spans(lo: int, hi: int, unit: int, step: int) -> list[tuple[int, int]]:
		first = (lo // unit) * unit
		out = []
		for a in range(first, hi, step):
			s = max(lo, a)
			e = min(hi, a + step)
			if s < e:
				out.append((s, e))
		return out

	z_spans = _spans(zs0, zs1, cz, step_z)
	y_spans = _spans(ys0, ys1, cy, step_y)
	x_spans = _spans(xs0, xs1, cx, step_x)
	return [
		(za, zb, ya, yb, xa, xb)
		for za, zb in z_spans
		for ya, yb in y_spans
		for xa, xb in x_spans
	]


def run(
	*,
	input_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
	workers: int = 0,
	base_shape: tuple[int, int, int] | None = None,
	selected_channels: Sequence[str] | None = None,
	preprocess_params_override: dict | None = None,
	defer_normal_pyramid: bool = False,
) -> dict[str, Path]:
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())

	a = open_zarr(input_path, threads=1)
	try:
		raw_shape = tuple(int(v) for v in a.shape)
		if len(raw_shape) == 4:
			is_4d = True
			c_in, z_full, y_full, x_full = raw_shape
		elif len(raw_shape) == 3:
			is_4d = False
			c_in = 1
			z_full, y_full, x_full = raw_shape
		else:
			raise ValueError(f"input must be (C,Z,Y,X) or (Z,Y,X), got shape={raw_shape}")
		if str(a.dtype) != "uint8":
			raise ValueError(f"input dtype must be uint8, got {a.dtype}")

		params = dict(a.attrs.get("preprocess_params", {}) or {})
		if preprocess_params_override:
			params.update(preprocess_params_override)
		scaledown = int(params.get("scaledown", 1) or 1)
		first_filled_level = _first_filled_level_from_downscale(scaledown)
		if first_filled_level >= int(levels):
			raise ValueError(
				f"--levels ({levels}) must exceed first_filled_level ({first_filled_level}) for scaledown={scaledown}"
			)

		zs0, ys0, xs0 = 0, 0, 0
		zs1, ys1, xs1 = z_full, y_full, x_full
		crop_xyzwhd = params.get("crop_xyzwhd", None)
		if isinstance(crop_xyzwhd, (list, tuple)) and len(crop_xyzwhd) == 6:
			x0f, y0f, z0f, wf, hf, df = (int(v) for v in crop_xyzwhd)
			if wf > 0 and hf > 0 and df > 0:
				xs0 = max(0, x0f // scaledown)
				ys0 = max(0, y0f // scaledown)
				zs0 = max(0, z0f // scaledown)
				xs1 = min(x_full, _ceil_div(x0f + wf, scaledown))
				ys1 = min(y_full, _ceil_div(y0f + hf, scaledown))
				zs1 = min(z_full, _ceil_div(z0f + df, scaledown))
		if zs1 <= zs0 or ys1 <= ys0 or xs1 <= xs0:
			raise ValueError(
				f"empty crop selection for conversion: z=[{zs0},{zs1}) y=[{ys0},{ys1}) x=[{xs0},{xs1})"
			)

		channels = _channels_from_src(c_in=c_in, params=params)
		if selected_channels is None:
			target_indices = list(range(len(channels)))
		else:
			wanted = list(dict.fromkeys(str(c) for c in selected_channels))
			missing = [c for c in wanted if c not in channels]
			if missing:
				raise KeyError(f"Requested channel(s) {missing} not in {channels} for {input_path}")
			target_indices = [channels.index(c) for c in wanted]

		target_names = [channels[ci] for ci in target_indices]
		ch_n = len(target_indices)
		local_normal_pair = (not defer_normal_pyramid) and {"nx", "ny"}.issubset(set(target_names))
		normal_pair_paths: dict[str, Path] = {}
		written_paths: dict[str, Path] = {}

		if base_shape is not None:
			base_full_shape = base_shape
		else:
			bs = params.get("base_shape_zyx")
			if bs is not None and len(bs) == 3:
				base_full_shape = tuple(int(v) for v in bs)
			else:
				base_full_shape = (z_full * scaledown, y_full * scaledown, x_full * scaledown) if scaledown > 1 else (z_full, y_full, x_full)
		crop_z = zs1 - zs0
		crop_y = ys1 - ys0
		crop_x = xs1 - xs0
		crop_zyx = (zs0, ys0, xs0, crop_z, crop_y, crop_x)

		TAG = "[convert_fit_zarr_to_vc3d_omezarr]"
		t0_all = time.time()

		for step_idx, ci in enumerate(target_indices):
			ch = channels[ci]
			out_path = Path(f"{output_prefix}_{ch}.ome.zarr")
			written_paths[ch] = out_path
			if local_normal_pair and ch in {"nx", "ny"}:
				normal_pair_paths[ch] = out_path
			g = create_group(out_path, overwrite=True)

			level_arrays: dict[int, ZarrWriter] = {}
			for lv in range(levels):
				sh = shape_div2(base_full_shape, lv)
				level_arrays[lv] = g.create_array(
					str(lv),
					shape=sh,
					chunks=(min(sh[0], chunk), min(sh[1], chunk), min(sh[2], chunk)),
					dtype=np.uint8,
					skip_empty=True,
				)

			base = level_arrays[first_filled_level]
			tiles = _aligned_crop_tiles(zs0, zs1, ys0, ys1, xs0, xs1, base.chunks)
			sz_gb = crop_z * crop_y * crop_x / 1e9
			print(
				f"{TAG} {ch}: {crop_z}×{crop_y}×{crop_x} ({sz_gb:.1f} Gvox), "
				f"{len(tiles)} tiles, {workers} workers, {levels} levels",
				flush=True,
			)
			t0 = time.time()

			def _write_tile(tile_bounds, ci=ci, base=base):
				za, zb, ya, yb, xa, xb = tile_bounds
				block = a[ci, za:zb, ya:yb, xa:xb] if is_4d else a[za:zb, ya:yb, xa:xb]
				if np.any(block):
					base[za:zb, ya:yb, xa:xb] = block

			_run_with_progress(tiles, _write_tile, workers=workers, prefix=f"{TAG} {ch} L{first_filled_level}")

			# --- Phase 2: Build coarser pyramid levels ---
			delay_this_normal = ch in {"nx", "ny"} and (defer_normal_pyramid or local_normal_pair)
			if delay_this_normal:
				print(f"{TAG} {ch}: delaying coarser levels for paired normal pyramid", flush=True)
			else:
				is_grad = ch == "grad_mag"
				build_scalar_omezarr_pyramid(
					out_path,
					first_filled_level,
					levels,
					workers=workers,
					crop_zyx=crop_zyx,
					label=ch,
					zero_overrides=is_grad,
					scan_existing_source_chunks=is_grad,
				)

			# --- Phase 3: Upsample stored chunks to fill levels below first_filled_level ---
			for lv in range(first_filled_level - 1, -1, -1):
				_upsample_level(
					src=level_arrays[lv + 1],
					dst=level_arrays[lv],
					chunk=chunk,
					workers=workers,
					prefix=f"{TAG} {ch} L{lv} (upsample)",
				)

			datasets = []
			for lv in range(levels):
				scale_fac = 2 ** lv
				datasets.append({
					"path": str(lv),
					"coordinateTransformations": [{
						"type": "scale",
						"scale": [float(scale_fac)] * 3,
					}],
				})
			g.attrs["multiscales"] = [{
				"version": "0.4",
				"name": str(ch),
				"axes": [
					{"name": "z", "type": "space", "unit": "pixel"},
					{"name": "y", "type": "space", "unit": "pixel"},
					{"name": "x", "type": "space", "unit": "pixel"},
				],
				"datasets": datasets,
			}]
			g.attrs["source_preprocess_params"] = params
			g.attrs["vc3d_convert"] = {
				"channel": str(ch),
				"channel_index": ci,
				"scaledown": scaledown,
				"crop_scaled_zyx_start": [zs0, ys0, xs0],
				"crop_scaled_zyx_stop": [zs1, ys1, xs1],
				"crop_offset_zyx": [zs0, ys0, xs0],
				"base_full_shape": list(base_full_shape),
				"levels": levels,
				"first_filled_level": first_filled_level,
			}

			elapsed_ch = time.time() - t0
			print(f"{TAG} {ch} done in {elapsed_ch:.1f}s", flush=True)

			done = step_idx + 1
			elapsed = max(1e-6, time.time() - t0_all)
			per = elapsed / done
			eta = max(0.0, per * (ch_n - done))
			eta_m = int(eta // 60.0)
			eta_s = int(eta % 60.0)
			bar_w = 30
			fill = int(round((done / max(1, ch_n)) * bar_w))
			bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
			print(
				f"\r[convert_fit_zarr_to_vc3d_omezarr] [{bar}] {done}/{ch_n} ({(100.0 * done / max(1, ch_n)):.1f}%) "
				f"eta {eta_m:02d}:{eta_s:02d}",
				end="",
				flush=True,
			)

		print("", flush=True)
		if local_normal_pair and {"nx", "ny"}.issubset(normal_pair_paths):
			print(f"{TAG} building paired nx/ny normal pyramid", flush=True)
			build_normal_omezarr_pyramid(
				normal_pair_paths["nx"],
				normal_pair_paths["ny"],
				first_filled_level,
				levels,
				workers=workers,
				crop_zyx=crop_zyx,
				label="normal",
				scan_existing_source_chunks=True,
			)
		return written_paths
	finally:
		a.close()


def run_from_manifest(
	*,
	manifest_path: str,
	output_prefix: str,
	levels: int,
	chunk: int,
	workers: int = 0,
	channels: list[str] | None = None,
	base_shape: tuple[int, int, int] | None = None,
) -> dict[str, Path]:
	"""Convert channels from a .lasagna.json or compact v3 .zarr manifest to VC3D OME-Zarr pyramids.

	Default channels: cos + pred_dt (if present), else all channels.
	base_shape from manifest's base_shape_zyx, or derived from finest zarr.
	"""
	try:
		from lasagna_volume import LasagnaVolume
	except ImportError:
		from lasagna.lasagna_volume import LasagnaVolume

	vol = LasagnaVolume.load(manifest_path)
	if not vol.groups:
		raise ValueError(f"Manifest has no channel groups: {manifest_path}")
	TAG = "[convert_fit_zarr_to_vc3d_omezarr]"

	if base_shape is None:
		base_shape = vol.base_shape_zyx
	if base_shape is None:
		min_sd = min(g.sd_fac for g in vol.groups.values())
		for g in vol.groups.values():
			if g.sd_fac == min_sd:
				zarr_path = str(vol.base_directory / g.zarr_path)
				a = open_zarr(zarr_path, threads=1)
				try:
					sh = tuple(int(v) for v in a.shape)
				finally:
					a.close()
				if len(sh) == 4:
					sh = sh[1:]
				base_shape = (sh[0] * min_sd, sh[1] * min_sd, sh[2] * min_sd)
				print(
					f"{TAG} derived base_shape={base_shape} from {g.zarr_path} (shape={sh}, sd_fac={min_sd})",
					flush=True,
				)
				break
	if base_shape is None:
		raise ValueError("Cannot determine base_shape: pass --base-shape or set base_shape_zyx in manifest")

	all_ch = vol.all_channels()
	if channels is None:
		selected = [name for name in ("cos", "pred_dt") if name in all_ch]
		if not selected:
			selected = list(all_ch)
	else:
		selected = list(dict.fromkeys(str(c) for c in channels))
	print(f"{TAG} manifest: {manifest_path}", flush=True)
	print(f"{TAG} base_shape={base_shape}, channels={selected}", flush=True)

	# Group requested channels by their underlying ChannelGroup so multi-channel stores open once.
	group_order: list[tuple[str, object, list[str]]] = []
	by_group_id: dict[int, list[str]] = {}
	group_names = {id(g): k for k, g in vol.groups.items()}
	for ch_name in selected:
		group, _ = vol.channel_group(ch_name)
		gid = id(group)
		if gid not in by_group_id:
			bucket: list[str] = []
			by_group_id[gid] = bucket
			group_order.append((group_names[gid], group, bucket))
		by_group_id[gid].append(ch_name)

	# Check whether nx and ny are both requested at the same scaledown across separate 3D groups.
	cross_group_normals = False
	normal_first_level: int | None = None
	if "nx" in selected and "ny" in selected:
		g_nx, _ = vol.channel_group("nx")
		g_ny, _ = vol.channel_group("ny")
		if g_nx is not g_ny and g_nx.sd_fac == g_ny.sd_fac:
			cross_group_normals = True
			normal_first_level = _first_filled_level_from_downscale(g_nx.sd_fac)

	written: dict[str, Path] = {}
	for g_key, group, req_channels in group_order:
		zarr_path = str(vol.base_directory / group.zarr_path)
		override: dict = {
			"scaledown": group.sd_fac,
			"channels": list(group.channels),
		}
		if vol.crops:
			override["crop_xyzwhd"] = list(vol.crops[0])
		defer_normals = cross_group_normals and bool({"nx", "ny"} & set(req_channels))
		print(
			f"{TAG} converting {req_channels} (group={g_key}, sd_fac={group.sd_fac})",
			flush=True,
		)
		paths = run(
			input_path=zarr_path,
			output_prefix=output_prefix,
			levels=levels,
			chunk=chunk,
			workers=workers,
			base_shape=base_shape,
			selected_channels=req_channels,
			preprocess_params_override=override,
			defer_normal_pyramid=defer_normals,
		)
		written.update(paths)

	if cross_group_normals and normal_first_level is not None and {"nx", "ny"}.issubset(written):
		print(f"{TAG} building paired nx/ny normal pyramid across manifest groups", flush=True)
		build_normal_omezarr_pyramid(
			written["nx"],
			written["ny"],
			normal_first_level,
			levels,
			workers=workers,
			label="normal",
			scan_existing_source_chunks=True,
		)

	return written


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Convert fit zarr (C,Z,Y,X or Z,Y,X uint8) or Lasagna manifest to per-channel VC3D OME-Zarr pyramids."
	)
	inp = p.add_mutually_exclusive_group(required=True)
	inp.add_argument("--input", help="Input zarr array path (C,Z,Y,X or Z,Y,X uint8).")
	inp.add_argument("--manifest", help="Input .lasagna.json or compact v3 .zarr manifest.")
	p.add_argument(
		"--output-prefix",
		required=True,
		help="Output prefix; files become <prefix>_<channel>.ome.zarr",
	)
	p.add_argument("--levels", type=int, default=5, help="Number of pyramid levels to create.")
	p.add_argument("--chunk", type=int, default=32, help="Chunk size in x/y/z.")
	p.add_argument(
		"--workers",
		type=int,
		default=0,
		help="Number of parallel workers (default: cpu_count).",
	)
	p.add_argument(
		"--base-shape",
		type=int,
		nargs=3,
		default=None,
		metavar=("Z", "Y", "X"),
		help="Explicit base (full-res) shape. Overrides manifest/zarr-derived shapes.",
	)
	p.add_argument(
		"--channels",
		nargs="+",
		default=None,
		help="Channel names to convert. Default in --manifest mode: cos + pred_dt (if present), else all.",
	)
	args = p.parse_args(argv)

	bs = tuple(args.base_shape) if args.base_shape else None

	if args.manifest:
		run_from_manifest(
			manifest_path=str(args.manifest),
			output_prefix=str(args.output_prefix),
			levels=int(args.levels),
			chunk=int(args.chunk),
			workers=int(args.workers),
			channels=args.channels,
			base_shape=bs,
		)
	else:
		run(
			input_path=str(args.input),
			output_prefix=str(args.output_prefix),
			levels=int(args.levels),
			chunk=int(args.chunk),
			workers=int(args.workers),
			base_shape=bs,
			selected_channels=args.channels,
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
