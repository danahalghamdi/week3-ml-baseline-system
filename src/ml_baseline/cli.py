from pathlib import Path
import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Week 3 ML baseline system CLI")

@app.command()
def make_sample_data(
    n_users: int = 50,
    seed: int = 42,
):
    """Generate sample feature data."""
    rng = np.random.default_rng(seed)

    user_id = [f"u{i:03d}" for i in range(1, n_users + 1)]
    country = rng.choice(["US", "CA", "GB"], size=n_users)
    n_orders = rng.integers(1, 10, size=n_users)
    avg_amount = rng.normal(10, 3, size=n_users).clip(min=1)
    total_amount = n_orders * avg_amount
    is_high_value = (total_amount >= 80).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "country": country,
            "n_orders": n_orders,
            "avg_amount": avg_amount.round(2),
            "total_amount": total_amount.round(2),
            "is_high_value": is_high_value,
        }
    )

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "features.csv", index=False)
    df.to_parquet(out / "features.parquet", index=False)

    typer.echo("features.csv and features.parquet written")

if __name__ == "__main__":
    app()