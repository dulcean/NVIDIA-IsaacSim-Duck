* Synchronize MicroDuck head-camera capture across worlds and keep UI reads from
  triggering extra renders. Batch visual tracking geometry, cache pixel grids and avoid
  physics-tick snapshot copies. Preserve shaded orange-ball pixels using color ratios,
  with a regression from real GPU frames. Reduce preview diagnostic transfers and report
  setup, playback, presentation and interrupted-run timings separately.
