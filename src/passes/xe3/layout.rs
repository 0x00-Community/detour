use mlirformat::Error;

use mlirpass::{Body, ImageLayout, OpId, pass};
use super::{Xe3Image};

#[pass("xe3-layout", on = "builtin.module")]
pub fn xe3_layout<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    ImageLayout(&Xe3Image).run(body, op)
}
