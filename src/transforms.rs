use crate::*;
use crate::Result;

// A `Transform` takes a `RegularTimeSeries` and the transform information in a `SeriesSpec` and
// outputs another `RegularTimeSeries`.  pub trait Transform {
pub trait Transform<D1: Date, V1: Value, D2: Date, V2: Value> {
    fn transform(time_series: RegularTimeSeries<D1, V1>) -> RegularTimeSeries<D2, V2>;
}

pub struct DropFirst<D1, V1>
where
    D1: Date,
    V1: Value,
{
    drop: usize,
    phantom_d: D1,
    phantom_v: V1,
}

pub trait UniformTransform<D1: Date, V1: Value> {
    fn uniform(&self, time_series: RegularTimeSeries<D1, V1>) -> Result<RegularTimeSeries<D1, V1>>;
}

impl<D1, V1> UniformTransform<D1, V1> for DropFirst<D1, V1>
where
    D1: Date,
    V1: Value
{
    fn uniform(&self, time_series: RegularTimeSeries<D1, V1>) -> Result<RegularTimeSeries<D1, V1>> {
        let ts: TimeSeries<D1, V1> = time_series.iter().skip(self.drop).collect();
        ts.try_into()
    }
}

