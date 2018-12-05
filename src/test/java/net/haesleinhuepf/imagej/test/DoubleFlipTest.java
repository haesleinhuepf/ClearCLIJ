package net.haesleinhuepf.imagej.test;

import clearcl.ClearCLImage;
import net.haesleinhuepf.imagej.ClearCLIJ;
import net.haesleinhuepf.imagej.kernels.Kernels;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.Duplicator;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.view.Views;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

/**
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * February 2018
 */
public class DoubleFlipTest
{
  public static void main(String... args) throws IOException
  {
    new ImageJ();
    new DoubleFlipTest().testFlipImageTwiceOnAllDevices();
  }

  @Test public void testFlipImageTwiceOnAllDevices() throws
                                                     IOException
  {
    for (String lDeviceName : ClearCLIJ.getAvailableDeviceNames())
    {
      ClearCLIJ lCLIJ = new ClearCLIJ(lDeviceName);

      System.out.println("Testing " + lDeviceName);

      ImagePlus
          lInputImagePlus =
          IJ.openImage("src/main/resources/flybrain.tif");

      ClearCLImage
          lCLImage =
          lCLIJ.converter(lInputImagePlus).getClearCLImage();

      RandomAccessibleInterval
          lInputImagePlus2 =
          lCLIJ.converter(lCLImage).getRandomAccessibleInterval();

      assertTrue(TestUtilities.compareIterableIntervals(Views.iterable(
          lInputImagePlus2),
                                                        ImageJFunctions.<UnsignedShortType>wrap(
                                                            lInputImagePlus)));

      RandomAccessibleInterval<UnsignedShortType>
          lInputImg =
          ImageJFunctions.wrap(lInputImagePlus);

      RandomAccessibleInterval<UnsignedShortType>
          lOutputImg =
          ImageJFunctions.wrap(new Duplicator().run(lInputImagePlus));

      ClearCLImage
          lSrcImage =
          lCLIJ.converter(lInputImg).getClearCLImage();
      ClearCLImage
          lDstImage =
          lCLIJ.converter(lOutputImg).getClearCLImage();

      // flip once
      Kernels.flip(lCLIJ, lSrcImage, lDstImage, true, false, false);

      // flip second time
      Map<String, Object> lParameterMap = new HashMap<>();
      lParameterMap.put("src", lDstImage);
      lParameterMap.put("dst", lSrcImage);
      lParameterMap.put("flipx", 1);
      lParameterMap.put("flipy", 0);
      lParameterMap.put("flipz", 0);

      lCLIJ.execute("src/main/java/clearcl/imagej/kernels/flip.cl",
                    "flip_3d",
                    lParameterMap);

      RandomAccessibleInterval<UnsignedShortType>
          lIntermediateResultImg =
          (RandomAccessibleInterval<UnsignedShortType>) lCLIJ.converter(
              lDstImage).getRandomAccessibleInterval();

      System.out.println("Should be different:");
      assertFalse(TestUtilities.compareIterableIntervals(Views.iterable(
          lIntermediateResultImg), Views.iterable(lInputImg)));

      RandomAccessibleInterval<UnsignedShortType>
          lResultImg =
          (RandomAccessibleInterval<UnsignedShortType>) lCLIJ.converter(
              lSrcImage).getRandomAccessibleInterval();

      System.out.println("Should be same:");
      assertTrue(TestUtilities.compareIterableIntervals(Views.iterable(
          lResultImg), Views.iterable(lInputImg)));
      ImageJFunctions.show(lResultImg, "res " + lDeviceName);
      ImageJFunctions.show(lInputImg, "inp " + lDeviceName);

      lCLImage.close();
      lSrcImage.close();
      lDstImage.close();
    }
  }
}