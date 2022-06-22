use crate::*;
use crate::impls::{Monthly, SingleF32};
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
    phantom_d: PhantomData<D1>,
    phantom_v: PhantomData<V1>,
}

impl DropFirst<Monthly, SingleF32>  {
    pub fn new(n: usize) -> Self {
        DropFirst {
            drop: n,
            phantom_d: PhantomData,
            phantom_v: PhantomData,
        }
    }
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

#[cfg(test)]
pub mod test {

    use crate::{RegularTimeSeries, TimeSeries};
    use crate::impls::{Monthly, SingleF32};
    use super::{DropFirst, UniformTransform};

    #[test]
    fn drop_should_work() {
        let csv = "
            2018-06-01, 1.2
            2018-07-01, 1.3
            2018-08-01, 1.4
            2018-09-01, 1.5
            2018-10-01, 1.6";

        dbg!();

        let ts = TimeSeries::<Monthly, SingleF32>::from_csv_str(csv).unwrap();

        dbg!(&ts);

        assert!(false);
        let rts: RegularTimeSeries<Monthly, SingleF32> = ts 
            .try_into()
            .unwrap();

        // let dropped = DropFirst::new(2).uniform(rts).unwrap();
    }
}

