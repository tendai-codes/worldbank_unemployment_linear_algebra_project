use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::env;

#[derive(Debug, Deserialize, Clone)]
struct PanelRow {
    country: String,
    country_code: String,
    year: i32,
    unemployment: Option<f64>,
    inflation: Option<f64>,
    gdp_growth: Option<f64>,
    life_expectancy: Option<f64>,
    population_growth: Option<f64>,
}

#[derive(Debug, Serialize)]
struct EngineeredRow {
    country: String,
    country_code: String,
    year: i32,

    unemployment: Option<f64>,
    inflation: Option<f64>,
    gdp_growth: Option<f64>,
    life_expectancy: Option<f64>,
    population_growth: Option<f64>,

    gdp_growth_next_year: Option<f64>,
    downturn_risk_next_year: Option<u8>,
    downturn_next_year: Option<u8>,

    unemployment_lag1: Option<f64>,
    inflation_lag1: Option<f64>,
    gdp_growth_lag1: Option<f64>,
    life_expectancy_lag1: Option<f64>,
    population_growth_lag1: Option<f64>,

    unemployment_lag2: Option<f64>,
    inflation_lag2: Option<f64>,
    gdp_growth_lag2: Option<f64>,
    life_expectancy_lag2: Option<f64>,
    population_growth_lag2: Option<f64>,

    unemployment_change_1y: Option<f64>,
    inflation_change_1y: Option<f64>,
    gdp_growth_change_1y: Option<f64>,
    life_expectancy_change_1y: Option<f64>,
    population_growth_change_1y: Option<f64>,

    unemployment_trend_3y: Option<f64>,
    inflation_trend_3y: Option<f64>,
    gdp_growth_trend_3y: Option<f64>,
    life_expectancy_trend_3y: Option<f64>,
    population_growth_trend_3y: Option<f64>,
}

#[derive(Clone, Copy)]
enum Feature {
    Unemployment,
    Inflation,
    GdpGrowth,
    LifeExpectancy,
    PopulationGrowth,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        return Err(anyhow!(
            "Usage: cargo run -- <input_csv> <output_csv>\nExample: cargo run -- ../data/worldbank_panel_final.csv ../data/worldbank_panel_engineered_rust.csv"
        ));
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let mut rows = read_panel_rows(input_path)?;
    sort_panel_rows(&mut rows);
    let engineered_rows = engineer_features(&rows);
    write_engineered_rows(output_path, &engineered_rows)?;

    println!("Read {} rows", rows.len());
    println!("Wrote {} engineered rows to {}", engineered_rows.len(), output_path);

    Ok(())
}

fn read_panel_rows(path: &str) -> Result<Vec<PanelRow>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Could not open input CSV: {}", path))?;

    let mut rows = Vec::new();
    for result in reader.deserialize() {
        let row: PanelRow = result.with_context(|| "Could not parse a CSV row")?;
        rows.push(row);
    }

    Ok(rows)
}

fn sort_panel_rows(rows: &mut [PanelRow]) {
    rows.sort_by(|a, b| match a.country_code.cmp(&b.country_code) {
        Ordering::Equal => a.year.cmp(&b.year),
        other => other,
    });
}

fn engineer_features(rows: &[PanelRow]) -> Vec<EngineeredRow> {
    let mut output = Vec::with_capacity(rows.len());

    for index in 0..rows.len() {
        let current = &rows[index];
        let lag1 = previous_same_country(rows, index, 1);
        let lag2 = previous_same_country(rows, index, 2);
        let next = next_same_country(rows, index);

        let gdp_growth_next_year = next.and_then(|row| row.gdp_growth);
        let downturn_risk_next_year = gdp_growth_next_year.map(|value| if value < 0.0 { 1 } else { 0 });

        output.push(EngineeredRow {
            country: current.country.clone(),
            country_code: current.country_code.clone(),
            year: current.year,

            unemployment: current.unemployment,
            inflation: current.inflation,
            gdp_growth: current.gdp_growth,
            life_expectancy: current.life_expectancy,
            population_growth: current.population_growth,

            gdp_growth_next_year,
            downturn_risk_next_year,
            downturn_next_year: downturn_risk_next_year,

            unemployment_lag1: lag1.and_then(|row| row.unemployment),
            inflation_lag1: lag1.and_then(|row| row.inflation),
            gdp_growth_lag1: lag1.and_then(|row| row.gdp_growth),
            life_expectancy_lag1: lag1.and_then(|row| row.life_expectancy),
            population_growth_lag1: lag1.and_then(|row| row.population_growth),

            unemployment_lag2: lag2.and_then(|row| row.unemployment),
            inflation_lag2: lag2.and_then(|row| row.inflation),
            gdp_growth_lag2: lag2.and_then(|row| row.gdp_growth),
            life_expectancy_lag2: lag2.and_then(|row| row.life_expectancy),
            population_growth_lag2: lag2.and_then(|row| row.population_growth),

            unemployment_change_1y: difference(current.unemployment, lag1.and_then(|row| row.unemployment)),
            inflation_change_1y: difference(current.inflation, lag1.and_then(|row| row.inflation)),
            gdp_growth_change_1y: difference(current.gdp_growth, lag1.and_then(|row| row.gdp_growth)),
            life_expectancy_change_1y: difference(current.life_expectancy, lag1.and_then(|row| row.life_expectancy)),
            population_growth_change_1y: difference(current.population_growth, lag1.and_then(|row| row.population_growth)),

            unemployment_trend_3y: three_year_slope(rows, index, Feature::Unemployment),
            inflation_trend_3y: three_year_slope(rows, index, Feature::Inflation),
            gdp_growth_trend_3y: three_year_slope(rows, index, Feature::GdpGrowth),
            life_expectancy_trend_3y: three_year_slope(rows, index, Feature::LifeExpectancy),
            population_growth_trend_3y: three_year_slope(rows, index, Feature::PopulationGrowth),
        });
    }

    output
}

fn previous_same_country(rows: &[PanelRow], index: usize, offset: usize) -> Option<&PanelRow> {
    if index < offset {
        return None;
    }

    let current = &rows[index];
    let previous = &rows[index - offset];

    if previous.country_code == current.country_code {
        Some(previous)
    } else {
        None
    }
}

fn next_same_country(rows: &[PanelRow], index: usize) -> Option<&PanelRow> {
    if index + 1 >= rows.len() {
        return None;
    }

    let current = &rows[index];
    let next = &rows[index + 1];

    if next.country_code == current.country_code {
        Some(next)
    } else {
        None
    }
}

fn difference(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    match (current, previous) {
        (Some(current_value), Some(previous_value)) => Some(current_value - previous_value),
        _ => None,
    }
}

fn three_year_slope(rows: &[PanelRow], index: usize, feature: Feature) -> Option<f64> {
    let row_1 = previous_same_country(rows, index, 2)?;
    let row_2 = previous_same_country(rows, index, 1)?;
    let row_3 = &rows[index];

    let values = [
        feature_value(row_1, feature)?,
        feature_value(row_2, feature)?,
        feature_value(row_3, feature)?,
    ];

    Some(least_squares_slope(&values))
}

fn feature_value(row: &PanelRow, feature: Feature) -> Option<f64> {
    match feature {
        Feature::Unemployment => row.unemployment,
        Feature::Inflation => row.inflation,
        Feature::GdpGrowth => row.gdp_growth,
        Feature::LifeExpectancy => row.life_expectancy,
        Feature::PopulationGrowth => row.population_growth,
    }
}

fn least_squares_slope(values: &[f64; 3]) -> f64 {
    let x_values = [0.0, 1.0, 2.0];
    let x_mean = 1.0;
    let y_mean = values.iter().sum::<f64>() / values.len() as f64;

    let numerator: f64 = x_values
        .iter()
        .zip(values.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();

    let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

    numerator / denominator
}

fn write_engineered_rows(path: &str, rows: &[EngineeredRow]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("Could not create output CSV: {}", path))?;

    for row in rows {
        writer.serialize(row).with_context(|| "Could not write CSV row")?;
    }

    writer.flush().with_context(|| "Could not flush output CSV")?;
    Ok(())
}
