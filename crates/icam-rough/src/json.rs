//! Just enough JSON for the reports' embedded data: numbers rounded to a fixed number of
//! decimals (non-finite as 0, or null where the page tells "none"), strings escaped,
//! objects and arrays from already written members.

/// A number to `d` decimals, trailing zeros trimmed; non-finite values as 0.
pub fn num(v: f64, d: usize) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    let s = format!("{v:.d$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if s == "-0" { "0".into() } else { s }
}

/// A number, or null when it is not finite.
pub fn opt(v: f64, d: usize) -> String {
    if v.is_finite() { num(v, d) } else { "null".into() }
}

pub fn arr(v: impl IntoIterator<Item = f64>, d: usize) -> String {
    let items: Vec<String> = v.into_iter().map(|x| num(x, d)).collect();
    format!("[{}]", items.join(","))
}

pub fn list(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(","))
}

pub fn text(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '<' => o.push_str("\\u003c"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// An object from (key, written value) pairs.
pub fn obj(members: &[(&str, String)]) -> String {
    let m: Vec<String> = members.iter().map(|(k, v)| format!("{}:{v}", text(k))).collect();
    format!("{{{}}}", m.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_text() {
        assert_eq!(num(1.2300, 3), "1.23");
        assert_eq!(num(-0.0004, 3), "0");
        assert_eq!(num(12.0, 0), "12");
        assert_eq!(num(f64::NAN, 2), "0");
        assert_eq!(opt(f64::INFINITY, 2), "null");
        assert_eq!(text("a \"b\" </script>"), "\"a \\\"b\\\" \\u003c/script>\"");
        assert_eq!(obj(&[("x", arr([1.0, 2.5], 1))]), "{\"x\":[1,2.5]}");
    }
}
