extern fn printi(a: u64) void;
extern fn printc(a: u8) void;
extern fn hlt() noreturn;
extern fn und() noreturn;

fn println(s: []const u8) void {
    for (s) |c| {
        printc(c);
    }
    printc('\n');
}

pub export fn kernel_main(pt: u64) noreturn {
    _ = pt;
    println("Paging is enabled!");
    // var page_table = PageTable(4).init(Frame.init(pt));
    // map_to(&page_table, VirtAddr.init(0xbaadf00d), Frame.init(0xcafeb000), 0b1);
    // println("0xbaadf00d =>");
    // const trans = translate(&page_table, VirtAddr.init(0xbaadf00d));
    // if (trans == null) {
    // println("Error translating 0xbaadf00d");
    // } else {
    // printi(@intCast(u64, trans.?));
    // }
    // hlt();
    hlt();
}
