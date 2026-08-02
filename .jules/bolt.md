# Bolt's Journal - Critical Learnings

## 2025-02-12 - [Fast Zero-Allocation String Parsing via Iterator Cloning]
**Learning:** Slicing substrings repeatedly like `&self.input[self.pos..]` in high-frequency loops (such as JSON parsing) incurs noticeable bounds-checking and UTF-8 decoding overhead. Cloning a Rust `std::str::Chars` iterator is extremely cheap (just copy 16 bytes of pointer metadata) and calling `.next()` is much faster. We can calculate the current byte position in $O(1)$ by subtracting the length of the remaining string slice from the total input length (`input.len() - chars.as_str().len()`). This is 100% safe, avoids all manual index tracking, and eliminates all string slicing bounds-checking.
**Action:** Use `std::str::Chars` iterator cloning and slice-subtraction length math next time a zero-allocation string parser or lexicographical scanner is optimized in Rust.
