from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    cors_origins: list[str]

    max_concurrent_runs: int = 2
    stream_poll_s: float = 1.0

    limit_max_drones: int = 10
    limit_max_points: int = 30
    limit_max_population: int | None = None
    limit_max_generations: int | None = None
    limit_max_grid_cells: int = 40000

    default_n_drones: int = 3
    default_n_points: int = 8
    default_population_size: int = 100
    default_generations: int = 100
    default_crossover_prob: float = 0.9
    default_crossover_eta: float = 15.0
    default_mutation_sigma: float = 0.05
    random_baseline_factor: float = 1.0

    battery_reserve: float = 0.15
    min_separation_m: float = 10.0
    base_clearance_m: float = 50.0
    separation_sampling_s: float = 2.0
    separation_max_samples: int = 4000

    visibility_open: float = 0.95
    visibility_brush: float = 0.70
    visibility_dense: float = 0.45

    priority_sigma_m: float = 200.0
    priority_path_weight: float = 0.5
    priority_path_width_m: float = 30.0


settings = Settings()


def visibility_factors() -> dict[str, float]:
    return {
        "open": settings.visibility_open,
        "brush": settings.visibility_brush,
        "dense": settings.visibility_dense,
    }
