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

const PAGE_SIZE: u64 = 4096;
const PHYS_OFFSET: u64 = 0xffff800000000000;
const KERNEL_OFFSET_PHYS: u64 = 0x200000;
const KERNEL_OFFSET_VIRT: u64 = KERNEL_OFFSET_PHYS | PHYS_OFFSET;
const KERNEL_END_PHYS: u64 = 0x500000;
const KERNEL_END_VIRT: u64 = KERNEL_END_PHYS | PHYS_OFFSET;
const ENTRIES_PER_TABLE: usize = 512; // each entry is a u64 (8 bytes)

const FrameAllocator = struct {
    const Self = @This();
    bump: u64,

    pub fn init(start: u64) Self {
        return Self{ .bump = start };
    }

    pub noinline fn alloc(self: *Self) u64 {
        const val = self.bump;
        self.bump += PAGE_SIZE;
        return val;
    }
};

const Frame = packed struct {
    const Self = @This();
    value: u64,

    pub fn init(value: u64) Self {
        return Self{ .value = value };
    }

    pub fn addr_value(self: Self) u64 {
        return self.value & 0x000f_ffff_ffff_f000;
    }
    pub fn pt_flags(self: Self) u64 {
        return self.value & 0x0fff;
    }
};

const VirtAddr = struct {
    const Self = @This();
    value: u64,

    pub fn init(value: u64) Self {
        return Self{ .value = value };
    }

    pub fn as_hhdm_phys(self: Self) u64 {
        return self.value - PHYS_OFFSET;
    }

    pub fn pt4_idx(self: Self) usize {
        return ((self.value / 4096) >> 27) & 0x1ff;
    }

    pub fn pt3_idx(self: Self) usize {
        return ((self.value / 4096) >> 18) & 0x1ff;
    }

    pub fn pt2_idx(self: Self) usize {
        return ((self.value / 4096) >> 9) & 0x1ff;
    }
    pub fn pt1_idx(self: Self) usize {
        return (self.value / 4096) & 0x1ff;
    }
    pub fn page_offset(self: Self) u64 {
        return self.value & 0xfff;
    }
};

pub fn PageTable(comptime Level: u8) type {
    return packed struct {
        const Self = @This();
        table: *[ENTRIES_PER_TABLE]Frame,

        pub noinline fn init(frame: Frame) Self {
            return Self{ .table = @intToPtr(*[ENTRIES_PER_TABLE]Frame, frame.addr_value() + PHYS_OFFSET) };
        }

        pub noinline fn next_table(self: *const Self, index: usize) ?PageTable(Level - 1) {
            if (self.table[index].value != 0) {
                return PageTable(Level - 1).init(self.table[index]);
            } else {
                return null;
            }
        }

        pub noinline fn next_table_create(self: *Self, index: usize, flags: u64, frame_alloc: *FrameAllocator) PageTable(Level - 1) {
            return self.next_table(index) orelse {
                const allocated = frame_alloc.alloc();
                const frame = Frame.init(allocated | flags | 0b1);
                self.table[index] = frame;
                return PageTable(Level - 1).init(frame);
            };
        }
    };
}

pub noinline fn translate(pt4: *PageTable(4), virt: VirtAddr) ?u64 {
    const pt3 = pt4.next_table(virt.pt4_idx()) orelse return null;
    const pt2 = pt3.next_table(virt.pt3_idx()) orelse return null;
    const pt1 = pt2.next_table(virt.pt2_idx()) orelse return null;
    const frame = pt1.table[virt.pt1_idx()];
    if (frame.pt_flags() & 0b1 == 0) {
        return null;
    }
    return frame.addr_value() + virt.page_offset();
}

pub noinline fn map_to(pt4: *PageTable(4), virt: VirtAddr, frame: Frame, flags: u64, frame_alloc: *FrameAllocator) void {
    var pt3 = pt4.next_table_create(virt.pt4_idx(), flags, frame_alloc);
    var pt2 = pt3.next_table_create(virt.pt3_idx(), flags, frame_alloc);
    var pt1 = pt2.next_table_create(virt.pt2_idx(), flags, frame_alloc);
    pt1.table[virt.pt1_idx()] = frame;
    pt1.table[virt.pt1_idx()].value |= flags | 0b1;
}

pub export fn kernel_main(pt: u64, fa_base: u64) noreturn {
    println("Paging is enabled!");
    var frame_alloc = FrameAllocator.init(fa_base);
    var page_table = PageTable(4).init(Frame.init(pt));
    map_to(&page_table, VirtAddr.init(0xbaadf00d), Frame.init(0xcafeb000), 0b1, &frame_alloc);
    println("0xbaadf00d =>");
    const trans = translate(&page_table, VirtAddr.init(0xbaadf00d));
    if (trans == null) {
        println("Error translating 0xbaadf00d");
    } else {
        printi(@intCast(u64, trans.?));
    }
    hlt();
}
