package net.haesleinhuepf.imagej.test;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import clearcl.enums.ImageChannelDataType;
import net.haesleinhuepf.imagej.ClearCLIJ;
import net.haesleinhuepf.imagej.kernels.Kernels;
import net.haesleinhuepf.imagej.utilities.ImageTypeConverter;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * This test goes through all available OpenCL devices and test if
 * type conversion works. Therefore, OffHeapPlanarStacks,
 * ClearCLImages and Imglib2 images are converted into each other.
 * Afterwards, it is checked if the images still contain the pixels
 * and are the same size.
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * February 2018
 */
public class ImageConverterTest
{
  ClearCLIJ mCLIJ = null;

  @Test public void testImgClearCLImageConverter()
  {
    for (String lDeviceName : ClearCLIJ.getAvailableDeviceNames())
    {
      mCLIJ = new ClearCLIJ(lDeviceName);
      System.out.println("Testing device " + lDeviceName);

      RandomAccessibleInterval<FloatType>
          lFloatImg =
          ArrayImgs.floats(new long[] { 5, 6, 7 });
      fillTestImage(lFloatImg);
      testBackAndForthConversionViaCLImage(lFloatImg);

      RandomAccessibleInterval<UnsignedByteType>
          lUnsignedByteImg =
          ArrayImgs.unsignedBytes(new long[] { 5, 6, 7 });
      fillTestImage(lUnsignedByteImg);
      testBackAndForthConversionViaCLImage(lUnsignedByteImg);

      RandomAccessibleInterval<ByteType>
          lByteImg =
          ArrayImgs.bytes(new long[] { 5, 6, 7 });
      fillTestImage(lByteImg);
      testBackAndForthConversionViaCLImage(lByteImg);

      RandomAccessibleInterval<UnsignedShortType>
          lUnsignedShortImg =
          ArrayImgs.unsignedShorts(new long[] { 5, 6, 7 });
      fillTestImage(lUnsignedShortImg);
      testBackAndForthConversionViaCLImage(lUnsignedShortImg);

      RandomAccessibleInterval<ShortType>
          lShortImg =
          ArrayImgs.shorts(new long[] { 5, 6, 7 });
      fillTestImage(lShortImg);
      testBackAndForthConversionViaCLImage(lShortImg);
    }
  }

  @Test public void testBufferConversion()
  {
    mCLIJ = new ClearCLIJ("HD");

    RandomAccessibleInterval<FloatType>
        lFloatImg =
        ArrayImgs.floats(new long[] { 5, 6, 7 });
    fillTestImage(lFloatImg);

    ClearCLIJ lCLIJ = mCLIJ;

    ClearCLBuffer
        lClearCLBuffer =
        lCLIJ.converter(lFloatImg).getClearCLBuffer();
    RandomAccessibleInterval
        lRAIconvertedTwice =
        lCLIJ.converter(lClearCLBuffer).getRandomAccessibleInterval();

    assertTrue(TestUtilities.compareIterableIntervals(Views.iterable(
        lFloatImg), Views.iterable(lRAIconvertedTwice)));

    lClearCLBuffer.close();
  }

  private <T extends RealType<T>> void fillTestImage(
      RandomAccessibleInterval<T> lFloatImg)
  {
    RandomAccess<T> lRandomAccess = lFloatImg.randomAccess();
    lRandomAccess.setPosition(new long[] { 1, 2, 3 });
    lRandomAccess.get().setReal(4);

    if (!(lRandomAccess.get() instanceof ByteType
          || lRandomAccess.get() instanceof UnsignedByteType))
    {
      lRandomAccess.setPosition(new long[] { 1, 3, 3 });
      lRandomAccess.get().setReal(40000);
    }
  }

  private <T extends RealType<T>> void testBackAndForthConversionViaCLImage(
      RandomAccessibleInterval<T> lRAI)
  {
    ClearCLIJ lCLIJ = mCLIJ;

    ClearCLImage
        lClearCLImage =
        lCLIJ.converter(lRAI).getClearCLImage();

    RandomAccessibleInterval<T>
        lRAIconvertedTwice =
        (RandomAccessibleInterval<T>) lCLIJ.converter(lClearCLImage)
                                           .getRandomAccessibleInterval();

    assertTrue(TestUtilities.compareIterableIntervals(Views.iterable(
        lRAI), Views.iterable(lRAIconvertedTwice)));

    lClearCLImage.close();
  }


  @Test public void convertHugeUnsignedShortImageTest()
  {
    mCLIJ = new ClearCLIJ("CPU");
    RandomAccessibleInterval<UnsignedShortType>
        lRAI =
        ArrayImgs.unsignedShorts(new long[] { 10, 10 });
    RandomAccess<UnsignedShortType> lRA = lRAI.randomAccess();
    lRA.setPosition(new long[] { 1, 1 });
    lRA.get().set(40000);


    testBackAndForthConversionViaCLImage(lRAI);

  }

  @Test public void convertHugeSignedShortImageTest()
  {
    mCLIJ = new ClearCLIJ("CPU");
    RandomAccessibleInterval<ShortType>
        lRAI =
        ArrayImgs.shorts(new long[] { 1, 1 });
    RandomAccess<ShortType> lRA = lRAI.randomAccess();
    lRA.setPosition(new long[] { 0, 0 });
    lRA.get().setReal(-25400);


    testBackAndForthConversionViaCLImage(lRAI);

  }

  @Test public void convertHugeFloatImageTest()
  {
    mCLIJ = new ClearCLIJ("CPU");
    RandomAccessibleInterval<FloatType>
        lRAI =
        ArrayImgs.floats(new long[] { 1, 1 });
    RandomAccess<FloatType> lRA = lRAI.randomAccess();
    lRA.setPosition(new long[] { 0, 0 });
    lRA.get().setReal(-25400);

    testBackAndForthConversionViaCLImage(lRAI);
  }


    @Test
    public void testConversionUnsignedShortStackToFloatCLImage() {
        mCLIJ = ClearCLIJ.getInstance();

        RandomAccessibleInterval<UnsignedShortType>
            rai =
            ArrayImgs.unsignedShorts(new long[] { 3, 3 });

        RandomAccess<UnsignedShortType> ra = rai.randomAccess();
        ra.setPosition(new int[] {1,1});
        ra.get().set(4);

        ClearCLImage stack = mCLIJ.converter(rai).getClearCLImage();

        // test starts here

        RandomAccessibleInterval rai2 = mCLIJ.converter(stack).getRandomAccessibleInterval();

        ClearCLImage clImage = mCLIJ.createCLImage(stack.getDimensions(), ImageChannelDataType.Float);
        ImageTypeConverter.copyRandomAccessibleIntervalToClearCLImage(rai2, clImage);

        ClearCLImage clImage2 = mCLIJ.createCLImage(stack.getDimensions(), ImageChannelDataType.UnsignedInt16);

        Kernels.copy(mCLIJ, clImage, clImage2);

        RandomAccessibleInterval<UnsignedShortType> rai3 = (RandomAccessibleInterval<UnsignedShortType>) mCLIJ.converter(clImage2).getRandomAccessibleInterval();

        assertTrue(TestUtilities.compareIterableIntervals(Views.iterable(rai2), Views.iterable(rai3)));

    }

}
