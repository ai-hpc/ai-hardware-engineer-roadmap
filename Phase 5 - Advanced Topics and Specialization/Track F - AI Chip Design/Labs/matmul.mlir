// A complete CPU exercise, tested with MLIR 20.1.2.
// C = A @ B = [[58, 64], [139, 154]]; return their checksum 415.
module {
  memref.global "private" constant @A : memref<2x3xf32> = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
  memref.global "private" constant @B : memref<3x2xf32> = dense<[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]>
  func.func @main() -> i32 {
    %a = memref.get_global @A : memref<2x3xf32>
    %b = memref.get_global @B : memref<3x2xf32>
    %c = memref.alloc() : memref<2x2xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<2x2xf32>)
    linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x2xf32>) outs(%c : memref<2x2xf32>)
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %c00 = memref.load %c[%i0, %i0] : memref<2x2xf32>
    %c01 = memref.load %c[%i0, %i1] : memref<2x2xf32>
    %c10 = memref.load %c[%i1, %i0] : memref<2x2xf32>
    %c11 = memref.load %c[%i1, %i1] : memref<2x2xf32>
    %s0 = arith.addf %c00, %c01 : f32
    %s1 = arith.addf %c10, %c11 : f32
    %sum = arith.addf %s0, %s1 : f32
    %checksum = arith.fptosi %sum : f32 to i32
    // Validate every element: equal checksums alone can hide wrong matrices.
    %e00 = arith.constant 58.0 : f32
    %e01 = arith.constant 64.0 : f32
    %e10 = arith.constant 139.0 : f32
    %e11 = arith.constant 154.0 : f32
    %ok00 = arith.cmpf oeq, %c00, %e00 : f32
    %ok01 = arith.cmpf oeq, %c01, %e01 : f32
    %ok10 = arith.cmpf oeq, %c10, %e10 : f32
    %ok11 = arith.cmpf oeq, %c11, %e11 : f32
    %ok0 = arith.andi %ok00, %ok01 : i1
    %ok1 = arith.andi %ok10, %ok11 : i1
    %ok = arith.andi %ok0, %ok1 : i1
    %failure = arith.constant -1 : i32
    %result = arith.select %ok, %checksum, %failure : i32
    memref.dealloc %c : memref<2x2xf32>
    return %result : i32
  }
}
