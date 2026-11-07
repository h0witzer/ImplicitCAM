//! The few complex operations the chatter limit needs.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const fn new(re: f64, im: f64) -> C {
        C { re, im }
    }

    /// A real times this.
    pub fn scale(self, k: f64) -> C {
        C::new(self.re * k, self.im * k)
    }

    /// Smith's division.
    pub fn div(self, o: C) -> C {
        let (a, b, c, d) = (self.re, self.im, o.re, o.im);
        if d.abs() < c.abs() {
            let doc = d / c;
            C::new((a + b * doc) / (c + d * doc), (b - a * doc) / (c + d * doc))
        } else {
            let cod = c / d;
            C::new((b + a * cod) / (d + c * cod), (-a + b * cod) / (d + c * cod))
        }
    }

    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Square root of a real number, as a complex one.
    pub fn sqrt_real(x: f64) -> C {
        if x < 0.0 { C::new(0.0, (-x).sqrt()) } else { C::new(x.sqrt(), 0.0) }
    }
}
