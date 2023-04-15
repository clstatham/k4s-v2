pub export fn demo(a: i64, b: *const i64) i64 {
    return a + b.*;
}
// start: zig or rust or C file

// -> compiled by zig / cargo (rustc) / clang into
//    platform-agnostic LLVM IR assembly / bitcode

// -> IR / BC is translated into k4sm assembly

// -> generated k4sm assembly files + human-written
//    k4sm assembly bootstrap are assembled into one
//    binary
