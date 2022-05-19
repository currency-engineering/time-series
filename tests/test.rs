use time_series::TimeSeries;
use time_series::impls::*;

#[test]
fn test_from_csv_with_a_file() {
    TimeSeries::<Monthly, SingleF32>::from_csv("./tests/test.csv", "%Y-%m-%d").unwrap();
}
