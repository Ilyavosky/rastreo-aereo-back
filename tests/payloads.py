DRONE = dict(
    name="Prueba",
    battery_wh=100,
    v_min=3,
    v_max=12,
    h_min=15,
    h_max=80,
    v_vertical=3,
    p_base_w=90,
    c_drag=0.02,
    c_altitude_j_m=30,
)
CAMERA = dict(
    name="Prueba",
    modality="optical",
    fov_deg=84,
    n_px_width=4000,
    power_w=8,
    battery_wh=20,
    max_range_m=120,
)
THERMAL = dict(
    name="Térmica",
    modality="thermal",
    fov_deg=45,
    n_px_width=640,
    power_w=5,
    battery_wh=15,
    max_range_m=80,
)
PERSON = dict(
    name="Prueba",
    posture="standing",
    critical_dim_m=0.5,
    px_min_optical=20,
    px_min_thermal=8,
)
SCENARIO_CONFIG = dict(
    width_m=400,
    height_m=400,
    cell_size_m=20,
    base=dict(x=20, y=20),
    last_seen=dict(x=250, y=250),
    vegetation_default="brush",
    vegetation_zones=[dict(type="dense", x0=100, y0=100, x1=300, y1=300)],
    forbidden_zones=[dict(x0=320, y0=320, x1=380, y1=380)],
    hills=[dict(x=200, y=200, height_m=30, sigma_m=80)],
    paths=[dict(kind="trail", points=[dict(x=20, y=20), dict(x=250, y=250)])],
)
SCENARIO = dict(name="Escenario de prueba", config=SCENARIO_CONFIG)
