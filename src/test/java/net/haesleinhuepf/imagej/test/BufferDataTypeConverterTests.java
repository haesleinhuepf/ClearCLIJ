package net.haesleinhuepf.imagej.test;

import clearcl.ClearCLBuffer;
import net.haesleinhuepf.imagej.ClearCLIJ;
import net.haesleinhuepf.imagej.kernels.Kernels;
import coremem.enums.NativeTypeEnum;
import org.junit.Test;

/**
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * April 2018
 */
public class BufferDataTypeConverterTests
{
  @Test
  public void testFloatToByteConversion() {
    ClearCLIJ clij = ClearCLIJ.getInstance();
    ClearCLBuffer bufferIn = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.Float);
    ClearCLBuffer bufferOut = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.Byte);

    Kernels.copy(clij, bufferIn, bufferOut);
  }


  @Test
  public void testFloatToUnsignedShortConversion() {
    ClearCLIJ clij = ClearCLIJ.getInstance();
    ClearCLBuffer bufferIn = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.Float);
    ClearCLBuffer bufferOut = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.UnsignedShort);

    Kernels.copy(clij, bufferIn, bufferOut);
  }


  @Test
  public void testUnsignedShortToByteConversion() {
    ClearCLIJ clij = ClearCLIJ.getInstance();
    ClearCLBuffer bufferIn = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.UnsignedShort);
    ClearCLBuffer bufferOut = clij.createCLBuffer(new long[]{10,10,10}, NativeTypeEnum.Byte);

    Kernels.copy(clij, bufferIn, bufferOut);
  }
}
