use time_series::{StringRecord, TimeSeries, ts_error, TSError};
use time_series::impls::*;

// Verify that an implementation compiles
#[derive(Copy, Clone)]
pub struct TestF32(pub f32);

impl TryFrom<StringRecord> for TestF32 {
    type Error = TSError;

    fn try_from(record: StringRecord) -> Result<Self, Self::Error> {
    
       if record.inner().len() != 1 { 
           return Err(TSError::Len("Expected records with a date and a single value.".to_owned()))
       }

       Ok(TestF32(
           record.inner().get(0)
                .ok_or(ts_error("Failed to get a singular value", Some(&record), None))?
                .parse()
                .map_err(|_| ts_error("Failed to parse data", Some(&record), None))?
        ))
    }
}

#[test]
fn test_from_csv_with_a_file() {
    TimeSeries::<Monthly, SingleF32>::from_csv("./tests/test.csv", "%Y-%m-%d").unwrap();
}
