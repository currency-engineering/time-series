use time_series::{
    TimeSeries
};
use time_series::date_impls::Monthly;

#[test]
fn test_from_csv_with_a_file() {
    TimeSeries::<Monthly, 1>::from_csv("./tests/test.csv", "%Y-%m-%d").unwrap();
}
