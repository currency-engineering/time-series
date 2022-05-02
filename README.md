The `time-series` crate constrains a data point to an fixed length array of floating point
numbers. Each data point is associated with a date. Dates are unusual in that they map to a
regular scale, so that monthly dates are always evenly separated.

#### Examples

```
// The standard procedure is to create a `TimeSeries` from `csv` data. `
// ::<1> defines the data array to be of length 1.
```
let ts = TimeSeries::<1>::from_csv("./tests/test.csv".into()).unwrap();
```

// And then to convert to a regular time-series with ordered data with regular
// intervals and no missing points.
```
let rts: RegularTimeSeries::<1> = ts.try_into().unwrap();
```

// When we create an iterator we define a range of dates to iterate over, or
// `None, None` for an open range.
```
let range = DateRange::new(None, Some(MonthlyDate::ym(2013,1)));
let iter = rts.iter(range);
```
