package net.haesleinhuepf.imagej.test;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import clearcl.enums.ImageChannelDataType;
import net.haesleinhuepf.imagej.ClearCLIJ;
import net.haesleinhuepf.imagej.kernels.Kernels;
import clearcl.util.ElapsedTime;
import coremem.enums.NativeTypeEnum;
import ij.*;
import ij.gui.NewImage;
import ij.gui.Roi;
import ij.plugin.Duplicator;
import ij.plugin.GaussianBlur3D;
import ij.plugin.ImageCalculator;
import ij.plugin.filter.MaximumFinder;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/*
 * Test class for all OpenCL kernels accessible via the Kernels class
 *
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * March 2018
 */
public class KernelsTest
{

  ImagePlus testFlyBrain3D;
  ImagePlus testFlyBrain2D;
  ImagePlus testImp1;
  ImagePlus testImp2;
  ImagePlus testImp2D1;
  ImagePlus testImp2D2;
  ImagePlus mask3d;
  ImagePlus mask2d;
  static ClearCLIJ clij;

  @Before public void initTest()
  {
    testFlyBrain3D = IJ.openImage("src/main/resources/flybrain.tif");
    testFlyBrain2D = new Duplicator().run(testFlyBrain3D, 1, 1);


    testImp1 =
        NewImage.createImage("",
                             100,
                             100,
                             12,
                             16,
                             NewImage.FILL_BLACK);
    testImp2 =
        NewImage.createImage("",
                             100,
                             100,
                             12,
                             16,
                             NewImage.FILL_BLACK);
    mask3d =
        NewImage.createImage("",
                             100,
                             100,
                             12,
                             16,
                             NewImage.FILL_BLACK);

    for (int z = 0; z < 5; z++)
    {
      testImp1.setZ(z + 1);
      ImageProcessor ip1 = testImp1.getProcessor();
      ip1.set(5, 5, 1);
      ip1.set(6, 6, 1);
      ip1.set(7, 7, 1);

      testImp2.setZ(z + 1);
      ImageProcessor ip2 = testImp2.getProcessor();
      ip2.set(7, 5, 2);
      ip2.set(6, 6, 2);
      ip2.set(5, 7, 2);

      if (z < 3)
      {
        mask3d.setZ(z + 3);
        ImageProcessor ip3 = mask3d.getProcessor();
        ip3.set(2, 2, 1);
        ip3.set(2, 3, 1);
        ip3.set(2, 4, 1);
        ip3.set(3, 2, 1);
        ip3.set(3, 3, 1);
        ip3.set(3, 4, 1);
        ip3.set(4, 2, 1);
        ip3.set(4, 3, 1);
        ip3.set(4, 4, 1);
      }
    }

    testImp2D1 = new Duplicator().run(testImp1, 1, 1);
    testImp2D2 = new Duplicator().run(testImp1, 1, 1);
    mask2d = new Duplicator().run(mask3d, 3, 3);

    if (clij == null)
    {
      clij = ClearCLIJ.getInstance();
          //new ClearCLIJ("Geforce");
      //new ClearCLIJ("HD");
    }
  }

  @Test public void absolute3d() {
    ImagePlus negativeImp =
            NewImage.createImage("",
                    2,
                    2,
                    2,
                    32,
                    NewImage.FILL_BLACK);

    ImageProcessor ip1 = negativeImp.getProcessor();
    ip1.setf(0, 1, -1.0f);

    ClearCLImage input = clij.converter(negativeImp).getClearCLImage();

    assertEquals(-1, Kernels.sumPixels(clij, input), 0.0001);

    ClearCLImage abs = clij.createCLImage(input);
    Kernels.absolute(clij, input, abs);
    assertEquals(1, Kernels.sumPixels(clij, abs),  0.0001);
  }

  @Test public void absolute3d_Buffer() {
    ImagePlus negativeImp =
            NewImage.createImage("",
                    2,
                    2,
                    2,
                    32,
                    NewImage.FILL_BLACK);

    ImageProcessor ip1 = negativeImp.getProcessor();
    ip1.setf(0, 1, -1.0f);

    ClearCLBuffer input = clij.converter(negativeImp).getClearCLBuffer();

    assertEquals(-1, Kernels.sumPixels(clij, input), 0.0001);

    ClearCLBuffer abs = clij.createCLBuffer(input);
    Kernels.absolute(clij, input, abs);
    assertEquals(1, Kernels.sumPixels(clij, abs),  0.0001);
  }

  @Test public void addPixelwise3d()
  {
    // do operation with ImageJ
    ImageCalculator ic = new ImageCalculator();
    ImagePlus sumImp = ic.run("Add create stack", testImp1, testImp2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.addPixelwise(clij, src, src1, dst);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }


  @Test public void addPixelwise3d_Buffers()
  {
    // do operation with ImageJ
    ImageCalculator ic = new ImageCalculator();
    ImagePlus sumImp = ic.run("Add create stack", testImp1, testImp2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp1).getClearCLBuffer();

    Kernels.addPixelwise(clij, src, src1, dst);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addPixelwise2d()
  {
    // do operation with ImageJ
    ImageCalculator ic = new ImageCalculator();
    ImagePlus sumImp = ic.run("Add create", testImp2D1, testImp2D2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2D2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.addPixelwise(clij, src, src1, dst);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addPixelwise2d_Buffer()
  {
    // do operation with ImageJ
    ImageCalculator ic = new ImageCalculator();
    ImagePlus sumImp = ic.run("Add create", testImp2D1, testImp2D2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2D2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp2D1).getClearCLBuffer();

    Kernels.addPixelwise(clij, src, src1, dst);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addScalar3d()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp1);
    IJ.run(added, "Add...", "value=1 stack");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.addScalar(clij, src, dst, 1f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }


  @Test public void addScalar3d_Buffer()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp1);
    IJ.run(added, "Add...", "value=1 stack");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp1).getClearCLBuffer();

    Kernels.addScalar(clij, src, dst, 1f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void addScalar2d()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp2D1);
    IJ.run(added, "Add...", "value=1");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.addScalar(clij, src, dst, 1f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void addScalar2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp2D1);
    IJ.run(added, "Add...", "value=1");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp2D1).getClearCLBuffer();

    Kernels.addScalar(clij, src, dst, 1f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void addWeightedPixelwise3d()
  {
    float factor1 = 3f;
    float factor2 = 2;

    // do operation with ImageJ
    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    ImagePlus testImp2copy = new Duplicator().run(testImp2);
    IJ.run(testImp1copy,
            "Multiply...",
            "value=" + factor1 + " stack");
    IJ.run(testImp2copy,
            "Multiply...",
            "value=" + factor2 + " stack");

    ImageCalculator ic = new ImageCalculator();
    ImagePlus
            sumImp =
            ic.run("Add create stack", testImp1copy, testImp2copy);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.addWeightedPixelwise(clij,
            src,
            src1,
            dst,
            factor1,
            factor2);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addWeightedPixelwise3d_Buffers()
  {
    float factor1 = 3f;
    float factor2 = 2;

    // do operation with ImageJ
    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    ImagePlus testImp2copy = new Duplicator().run(testImp2);
    IJ.run(testImp1copy,
            "Multiply...",
            "value=" + factor1 + " stack");
    IJ.run(testImp2copy,
            "Multiply...",
            "value=" + factor2 + " stack");

    ImageCalculator ic = new ImageCalculator();
    ImagePlus
            sumImp =
            ic.run("Add create stack", testImp1copy, testImp2copy);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp1).getClearCLBuffer();

    Kernels.addWeightedPixelwise(clij,
            src,
            src1,
            dst,
            factor1,
            factor2);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addWeightedPixelwise2d()
  {
    float factor1 = 3f;
    float factor2 = 2;

    // do operation with ImageJ
    ImagePlus testImp1copy = new Duplicator().run(testImp2D1);
    ImagePlus testImp2copy = new Duplicator().run(testImp2D2);
    IJ.run(testImp1copy, "Multiply...", "value=" + factor1 + " ");
    IJ.run(testImp2copy, "Multiply...", "value=" + factor2 + " ");

    ImageCalculator ic = new ImageCalculator();
    ImagePlus
        sumImp =
        ic.run("Add create ", testImp1copy, testImp2copy);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2D2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.addWeightedPixelwise(clij,
                                 src,
                                 src1,
                                 dst,
                                 factor1,
                                 factor2);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void addWeightedPixelwise2d_Buffers()
  {
    float factor1 = 3f;
    float factor2 = 2;

    // do operation with ImageJ
    ImagePlus testImp1copy = new Duplicator().run(testImp2D1);
    ImagePlus testImp2copy = new Duplicator().run(testImp2D2);
    IJ.run(testImp1copy, "Multiply...", "value=" + factor1 + " ");
    IJ.run(testImp2copy, "Multiply...", "value=" + factor2 + " ");

    ImageCalculator ic = new ImageCalculator();
    ImagePlus
            sumImp =
            ic.run("Add create ", testImp1copy, testImp2copy);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2D2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp2D1).getClearCLBuffer();

    Kernels.addWeightedPixelwise(clij,
            src,
            src1,
            dst,
            factor1,
            factor2);
    ImagePlus sumImpFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(sumImp, sumImpFromCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void argMaxProjection()
  {
    // do operation with ImageJ
    ImagePlus
        maxProjection =
        NewImage.createShortImage("",
                                  testImp1.getWidth(),
                                  testImp2.getHeight(),
                                  1,
                                  NewImage.FILL_BLACK);
    ImageProcessor ipMax = maxProjection.getProcessor();
    ImagePlus
        argMaxProjection =
        NewImage.createShortImage("",
                                  testImp1.getWidth(),
                                  testImp2.getHeight(),
                                  1,
                                  NewImage.FILL_BLACK);
    ImageProcessor ipArgMax = maxProjection.getProcessor();

    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    for (int z = 0; z < testImp1copy.getNSlices(); z++)
    {
      testImp1copy.setZ(z + 1);
      ImageProcessor ip = testImp1copy.getProcessor();
      for (int x = 0; x < testImp1copy.getWidth(); x++)
      {
        for (int y = 0; y < testImp1copy.getHeight(); y++)
        {
          float value = ip.getf(x, y);
          if (value > ipMax.getf(x, y))
          {
            ipMax.setf(x, y, value);
            ipArgMax.setf(x, y, z);
          }
        }
      }
    }

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(new long[] { src.getWidth(),
                                        src.getHeight() },
                           src.getChannelDataType());
    ClearCLImage
        dst_arg =
        clij.createCLImage(new long[] { src.getWidth(),
                                        src.getHeight() },
                           src.getChannelDataType());

    Kernels.argMaxProjection(clij, src, dst, dst_arg);

    ImagePlus maxProjectionCL = clij.converter(dst).getImagePlus();
    ImagePlus
        argMaxProjectionCL =
        clij.converter(dst_arg).getImagePlus();

    assertTrue(TestUtilities.compareImages(maxProjection,
                                           maxProjectionCL));
    assertTrue(TestUtilities.compareImages(argMaxProjection,
                                           argMaxProjectionCL));

    src.close();
    dst.close();
    dst_arg.close();
  }

  @Test public void argMaxProjection_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            maxProjection =
            NewImage.createShortImage("",
                    testImp1.getWidth(),
                    testImp2.getHeight(),
                    1,
                    NewImage.FILL_BLACK);
    ImageProcessor ipMax = maxProjection.getProcessor();
    ImagePlus
            argMaxProjection =
            NewImage.createShortImage("",
                    testImp1.getWidth(),
                    testImp2.getHeight(),
                    1,
                    NewImage.FILL_BLACK);
    ImageProcessor ipArgMax = maxProjection.getProcessor();

    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    for (int z = 0; z < testImp1copy.getNSlices(); z++)
    {
      testImp1copy.setZ(z + 1);
      ImageProcessor ip = testImp1copy.getProcessor();
      for (int x = 0; x < testImp1copy.getWidth(); x++)
      {
        for (int y = 0; y < testImp1copy.getHeight(); y++)
        {
          float value = ip.getf(x, y);
          if (value > ipMax.getf(x, y))
          {
            ipMax.setf(x, y, value);
            ipArgMax.setf(x, y, z);
          }
        }
      }
    }

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer
            dst =
            clij.createCLBuffer(new long[] { src.getWidth(),
                            src.getHeight() },
                    src.getNativeType());
    ClearCLBuffer
            dst_arg =
            clij.createCLBuffer(new long[] { src.getWidth(),
                            src.getHeight() },
                    src.getNativeType());

    Kernels.argMaxProjection(clij, src, dst, dst_arg);

    ImagePlus maxProjectionCL = clij.converter(dst).getImagePlus();
    ImagePlus
            argMaxProjectionCL =
            clij.converter(dst_arg).getImagePlus();

    assertTrue(TestUtilities.compareImages(maxProjection,
            maxProjectionCL));
    assertTrue(TestUtilities.compareImages(argMaxProjection,
            argMaxProjectionCL));

    src.close();
    dst.close();
    dst_arg.close();
  }

  @Test public void binaryAnd2d() {
    ClearCLImage clearCLImage = clij.converter(mask2d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());
    ClearCLImage clearCLImageAnd  = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryAnd(clij, clearCLImage, clearCLImageNot, clearCLImageAnd);

    long numberOfPixelsAnd = (long)Kernels.sumPixels(clij, clearCLImageAnd);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixelsAnd, 0);
    assertEquals(clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth(), numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageAnd.close();
  }

  @Test public void binaryAnd2d_Buffers() {
    ClearCLBuffer clearCLImage = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());
    ClearCLBuffer clearCLImageAnd  = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryAnd(clij, clearCLImage, clearCLImageNot, clearCLImageAnd);

    long numberOfPixelsAnd = (long)Kernels.sumPixels(clij, clearCLImageAnd);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixelsAnd, 0);
    assertEquals(clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth(), numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageAnd.close();
  }

  @Test public void binaryAnd3d() {
    ClearCLImage clearCLImage = clij.converter(mask3d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());
    ClearCLImage clearCLImageAnd  = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryAnd(clij, clearCLImage, clearCLImageNot, clearCLImageAnd);

    long numberOfPixelsAnd = (long)Kernels.sumPixels(clij, clearCLImageAnd);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixelsAnd, 0);
    assertEquals(clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth(), numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageAnd.close();
  }


  @Test public void binaryAnd3d_Buffers() {
    ClearCLBuffer clearCLImage = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());
    ClearCLBuffer clearCLImageAnd  = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryAnd(clij, clearCLImage, clearCLImageNot, clearCLImageAnd);

    long numberOfPixelsAnd = (long)Kernels.sumPixels(clij, clearCLImageAnd);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixelsAnd, 0);
    assertEquals(clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth(), numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageAnd.close();
  }

  @Test public void binaryNot2d() throws InterruptedException {
    ClearCLImage clearCLImage = clij.converter(mask2d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);

    long numberOfPixels = clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth();

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
  }

  @Test public void binaryNot2d_Buffers() throws InterruptedException {
    ClearCLBuffer clearCLImage = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);

    long numberOfPixels = clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth();

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
  }

  @Test public void binaryNot3d() throws InterruptedException {
    ClearCLImage clearCLImage = clij.converter(mask3d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);

    long numberOfPixels = clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth();

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
  }


  @Test public void binaryNot3d_Buffers() throws InterruptedException {
    ClearCLBuffer clearCLImage = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);

    long numberOfPixels = clearCLImage.getWidth() * clearCLImage.getHeight() * clearCLImage.getDepth();

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
  }

  @Test public void binaryOr2d() {
    ClearCLImage clearCLImage = clij.converter(mask2d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());
    ClearCLImage clearCLImageOr  = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryOr(clij, clearCLImage, clearCLImageNot, clearCLImageOr);

    long numberOfPixels = (long)Kernels.sumPixels(clij, clearCLImageOr);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageOr.close();
  }

  @Test public void binaryOr2d_Buffer() {
    ClearCLBuffer clearCLImage = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());
    ClearCLBuffer clearCLImageOr  = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryOr(clij, clearCLImage, clearCLImageNot, clearCLImageOr);

    long numberOfPixels = (long)Kernels.sumPixels(clij, clearCLImageOr);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageOr.close();
  }

  @Test public void binaryOr3d() {
    ClearCLImage clearCLImage = clij.converter(mask3d).getClearCLImage();
    ClearCLImage clearCLImageNot = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());
    ClearCLImage clearCLImageOr  = clij.createCLImage(clearCLImage.getDimensions(), clearCLImage.getChannelDataType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryOr(clij, clearCLImage, clearCLImageNot, clearCLImageOr);

    long numberOfPixels = (long)Kernels.sumPixels(clij, clearCLImageOr);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageOr.close();
  }

  @Test public void binaryOr3d_Buffers() {
    ClearCLBuffer clearCLImage = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer clearCLImageNot = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());
    ClearCLBuffer clearCLImageOr  = clij.createCLBuffer(clearCLImage.getDimensions(), clearCLImage.getNativeType());

    Kernels.binaryNot(clij, clearCLImage, clearCLImageNot);
    Kernels.binaryOr(clij, clearCLImage, clearCLImageNot, clearCLImageOr);

    long numberOfPixels = (long)Kernels.sumPixels(clij, clearCLImageOr);

    long numberOfPositivePixels = (long)Kernels.sumPixels(clij, clearCLImage);
    long numberOfNegativePixels = (long)Kernels.sumPixels(clij, clearCLImageNot);

    assertEquals(numberOfPixels, numberOfNegativePixels + numberOfPositivePixels);
    clearCLImage.close();
    clearCLImageNot.close();
    clearCLImageOr.close();
  }

  @Test public void blur3d()
  {
    // do operation with ImageJ
    ImagePlus gauss = new Duplicator().run(testImp1);
    GaussianBlur3D.blur(gauss, 2, 2, 2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.blur(clij, src, dst, 6, 6, 6, 2f, 2f, 2f);
    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL));

    src.close();
    dst.close();
  }

  @Test public void blur3d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus gauss = new Duplicator().run(testImp1);
    GaussianBlur3D.blur(gauss, 2, 2, 2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.blur(clij, src, dst, 6, 6, 6, 2f, 2f, 2f);
    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL));

    src.close();
    dst.close();
  }

  @Test
  public void blur3dSeparable() throws InterruptedException {
    ClearCLImage src = clij.converter(testFlyBrain3D).getClearCLImage();
    ClearCLImage dstBlur = clij.createCLImage(src);
    ClearCLImage dstBlurSeparable = clij.createCLImage(src);

    Kernels.blur(clij, src, dstBlur, 15, 15, 15, 2f, 2f, 2f);
    Kernels.blurSeparable(clij, src, dstBlurSeparable, 2, 2, 2);

    //ClearCLBuffer srcB = clij.converter(testFlyBrain3D).getClearCLBuffer();
    //ClearCLBuffer dstBlurSeparableB = clij.createCLBuffer(srcB);
    //Kernels.blurSeparable(clij, srcB, dstBlurSeparableB, 2, 2, 2);

    ImagePlus gaussBlur = clij.converter(dstBlur).getImagePlus();
    ImagePlus gaussBlurSeparable = clij.converter(dstBlurSeparable).getImagePlus();
    //ImagePlus gaussBlurSeparableB = clij.converter(dstBlurSeparableB).getImagePlus();

    src.close();
    dstBlur.close();
    dstBlurSeparable.close();
    //srcB.close();
    //dstBlurSeparableB.close();

    assertTrue(TestUtilities.compareImages(gaussBlurSeparable, gaussBlur, 2.0));
    //assertTrue(TestUtilities.compareImages(gaussBlurSeparable, gaussBlurSeparableB, 2.0));
  }

//  @Test
//  public void blur3dSeparable_Buffer() throws InterruptedException {
//    ClearCLBuffer src = clij.converter(testFlyBrain3D).getClearCLBuffer();
//    ClearCLBuffer dstBlur = clij.createCLBuffer(src);
//    ClearCLBuffer dstBlurSeparable = clij.createCLBuffer(src);
//
//    Kernels.blur(clij, src, dstBlur, 15, 15, 15, 2, 2, 2);
//    Kernels.blurSeparable(clij, src, dstBlurSeparable, 2, 2, 2);
//
//    //ClearCLBuffer srcB = clij.converter(testFlyBrain3D).getClearCLBuffer();
//    //ClearCLBuffer dstBlurSeparableB = clij.createCLBuffer(srcB);
//    //Kernels.blurSeparable(clij, srcB, dstBlurSeparableB, 2, 2, 2);
//
//    ImagePlus gaussBlur = clij.converter(dstBlur).getImagePlus();
//    ImagePlus gaussBlurSeparable = clij.converter(dstBlurSeparable).getImagePlus();
//    //ImagePlus gaussBlurSeparableB = clij.converter(dstBlurSeparableB).getImagePlus();
//
//    src.close();
//    dstBlur.close();
//    dstBlurSeparable.close();
//    //srcB.close();
//    //dstBlurSeparableB.close();
//
//    assertTrue(TestUtilities.compareImages(gaussBlurSeparable, gaussBlur, 2.0));
//    //assertTrue(TestUtilities.compareImages(gaussBlurSeparable, gaussBlurSeparableB, 2.0));
//  }

  @Test public void blur2d()
  {
    // do operation with ImageJ
    ImagePlus gauss = new Duplicator().run(testFlyBrain2D);
    ImagePlus gaussCopy = new Duplicator().run(testFlyBrain2D);
    IJ.run(gauss, "Gaussian Blur...", "sigma=2");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(gaussCopy).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.blur(clij, src, dst, 15, 15, 2f, 2f);
    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL, 2));

    src.close();
    dst.close();
  }

  @Test public void blur2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus gauss = new Duplicator().run(testFlyBrain2D);
    ImagePlus gaussCopy = new Duplicator().run(testFlyBrain2D);
    IJ.run(gauss, "Gaussian Blur...", "sigma=2");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(gaussCopy).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.blur(clij, src, dst, 15, 15, 2f, 2f);
    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL, 2));

    src.close();
    dst.close();
  }

  @Test public void blurSliceBySlice()
  {
    // do operation with ImageJ
    ImagePlus gauss = new Duplicator().run(testFlyBrain3D);
    IJ.run(gauss, "Gaussian Blur...", "sigma=2 stack");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testFlyBrain3D).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.blurSliceBySlice(clij, src, dst, 15, 15, 2f, 2f);
    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL, 2));

    src.close();
    dst.close();
  }

//  @Test public void blurSliceBySlice_Buffers()
//  {
//    // do operation with ImageJ
//    ImagePlus gauss = new Duplicator().run(testFlyBrain3D);
//    IJ.run(gauss, "Gaussian Blur...", "sigma=2 stack");
//
//    // do operation with ClearCL
//    ClearCLBuffer src = clij.converter(testFlyBrain3D).getClearCLBuffer();
//    ClearCLBuffer dst = clij.createCLBuffer(src);
//
//    Kernels.blurSliceBySlice(clij, src, dst, 15, 15, 2, 2);
//    ImagePlus gaussFromCL = clij.converter(dst).getImagePlus();
//
//    assertTrue(TestUtilities.compareImages(gauss, gaussFromCL, 2));
//
//    src.close();
//    dst.close();
//  }

  @Test public void copyImageToBuffer3d()
  {
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(src.getDimensions(), src.getNativeType());

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copyImageToBuffer2d()
  {
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(src.getDimensions(), src.getNativeType());

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp2D1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copyBufferToImage3d()
  {
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.set(clij, dst, 0f);

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copyBufferToImage2d()
  {
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.set(clij, dst, 0f);

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp2D1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copy3d()
  {
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(src.getDimensions(),
                           src.getChannelDataType());
    ;

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copy2d()
  {
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(src.getDimensions(),
                           src.getChannelDataType());

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp2D1, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void copyBuffer3d()
  {
    ImagePlus imp = new Duplicator().run(testImp1);
    Img<FloatType> img = ImageJFunctions.convertFloat(testImp1);

    ClearCLBuffer src = clij.converter(imp).getClearCLBuffer();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(src.getDimensions(), src.getNativeType());

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();
    assertTrue(TestUtilities.compareImages(testImp1, copyFromCL));

    RandomAccessibleInterval
        rai =
        clij.converter(dst).getRandomAccessibleInterval();
    assertTrue(TestUtilities.compareIterableIntervals(img,
                                                      Views.iterable(
                                                          rai)));

    src.close();
    dst.close();
  }

  @Test public void copyBuffer2d()
  {
    ImagePlus imp = new Duplicator().run(testImp2D1);
    Img<FloatType> img = ImageJFunctions.convertFloat(testImp2D1);

    ClearCLBuffer src = clij.converter(imp).getClearCLBuffer();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(src.getDimensions(), src.getNativeType());

    Kernels.copy(clij, src, dst);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();
    assertTrue(TestUtilities.compareImages(testImp2D1, copyFromCL));

    RandomAccessibleInterval
        rai =
        clij.converter(dst).getRandomAccessibleInterval();
    assertTrue(TestUtilities.compareIterableIntervals(img,
                                                      Views.iterable(
                                                          rai)));

    src.close();
    dst.close();
  }

  @Test public void copySlice()
  {
    // do operation with ImageJ
    ImagePlus copy = new Duplicator().run(testImp1, 3, 3);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(new long[] { src.getWidth(),
                                        src.getHeight() },
                           src.getChannelDataType());

    Kernels.copySlice(clij, src, dst, 2);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(copy, copyFromCL));

    // also test if putSliceInStack works
    Kernels.copySlice(clij, dst, src, 2);

    src.close();
    dst.close();
  }

  @Test public void copySliceBuffer()
  {
    // do operation with ImageJ
    ImagePlus copy = new Duplicator().run(testImp1, 3, 3);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(new long[] { src.getWidth(),
                                         src.getHeight() },
                            src.getNativeType());

    Kernels.copySlice(clij, src, dst, 2);
    ImagePlus copyFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(copy, copyFromCL));

    src.close();
    dst.close();
  }

  @Test public void crop3d()
  {
    // do operation with ImageJ
    Roi roi = new Roi(2, 2, 10, 10);
    testImp1.setRoi(roi);
    ImagePlus crop = new Duplicator().run(testImp1, 3, 12);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage
            dst =
            clij.createCLImage(new long[] { 10, 10, 10 },
                    src.getChannelDataType());

    Kernels.crop(clij, src, dst, 2, 2, 2);
    ImagePlus cropFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(crop, cropFromCL));

    src.close();
    dst.close();
  }

  @Test public void crop2d()
  {
    // do operation with ImageJ
    Roi roi = new Roi(2, 2, 10, 10);
    testImp2D1.setRoi(roi);
    ImagePlus crop = new Duplicator().run(testImp2D1);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(new long[] { 10, 10 },
                           src.getChannelDataType());

    Kernels.crop(clij, src, dst, 2, 2);
    ImagePlus cropFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(crop, cropFromCL));

    src.close();
    dst.close();
  }

  @Test public void cropBuffer3d()
  {
    // do operation with ImageJ
    Roi roi = new Roi(2, 2, 10, 10);
    testImp1.setRoi(roi);
    ImagePlus crop = new Duplicator().run(testImp1, 3, 12);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(new long[] { 10, 10, 10 },
                            src.getNativeType());

    Kernels.crop(clij, src, dst, 2, 2, 2);
    ImagePlus cropFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(crop, cropFromCL));

    src.close();
    dst.close();
  }

  @Test public void cropBuffer2d()
  {
    // do operation with ImageJ
    Roi roi = new Roi(2, 2, 10, 10);
    testImp2D1.setRoi(roi);
    ImagePlus crop = new Duplicator().run(testImp2D1);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer
        dst =
        clij.createCLBuffer(new long[] { 10, 10 },
                            src.getNativeType());

    Kernels.crop(clij, src, dst, 2, 2);
    ImagePlus cropFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(crop, cropFromCL));

    src.close();
    dst.close();
  }

  @Test public void detectMaxima3d()
  {
    // do operation with ImageJ
    ImagePlus
        spotsImage =
        NewImage.createImage("",
                             100,
                             100,
                             3,
                             16,
                             NewImage.FILL_BLACK);

    spotsImage.setZ(2);
    ImageProcessor ip1 = spotsImage.getProcessor();
    ip1.set(50, 50, 10);
    ip1.set(60, 60, 10);
    ip1.set(70, 70, 10);

    spotsImage.show();
    //IJ.run(spotsImage, "Find Maxima...", "noise=2 output=[Single Points]");

    ByteProcessor
        byteProcessor =
        new MaximumFinder().findMaxima(spotsImage.getProcessor(),
                                       2,
                                       MaximumFinder.SINGLE_POINTS,
                                       true);
    ImagePlus maximaImp = new ImagePlus("A", byteProcessor);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(spotsImage).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.detectOptima(clij, src, dst, 1, true);
    ImagePlus maximaFromCL = clij.converter(dst).getImagePlus();
    maximaFromCL = new Duplicator().run(maximaFromCL, 2, 2);

    IJ.run(maximaImp, "Divide...", "value=255");

    assertTrue(TestUtilities.compareImages(maximaImp, maximaFromCL));

    src.close();
    dst.close();
  }

  @Test public void detectMaxima3d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            spotsImage =
            NewImage.createImage("",
                    100,
                    100,
                    3,
                    16,
                    NewImage.FILL_BLACK);

    spotsImage.setZ(2);
    ImageProcessor ip1 = spotsImage.getProcessor();
    ip1.set(50, 50, 10);
    ip1.set(60, 60, 10);
    ip1.set(70, 70, 10);

    spotsImage.show();
    //IJ.run(spotsImage, "Find Maxima...", "noise=2 output=[Single Points]");

    ByteProcessor
            byteProcessor =
            new MaximumFinder().findMaxima(spotsImage.getProcessor(),
                    2,
                    MaximumFinder.SINGLE_POINTS,
                    true);
    ImagePlus maximaImp = new ImagePlus("A", byteProcessor);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(spotsImage).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.detectOptima(clij, src, dst, 1, true);
    ImagePlus maximaFromCL = clij.converter(dst).getImagePlus();
    maximaFromCL = new Duplicator().run(maximaFromCL, 2, 2);

    IJ.run(maximaImp, "Divide...", "value=255");

    assertTrue(TestUtilities.compareImages(maximaImp, maximaFromCL));

    src.close();
    dst.close();
  }

  @Test public void detectMaxima2d()
  {
    // do operation with ImageJ
    ImagePlus
        spotsImage =
        NewImage.createImage("",
                             100,
                             100,
                             1,
                             16,
                             NewImage.FILL_BLACK);

    spotsImage.setZ(1);
    ImageProcessor ip1 = spotsImage.getProcessor();
    ip1.set(50, 50, 10);
    ip1.set(60, 60, 10);
    ip1.set(70, 70, 10);

    spotsImage.show();
    //IJ.run(spotsImage, "Find Maxima...", "noise=2 output=[Single Points]");

    ByteProcessor
        byteProcessor =
        new MaximumFinder().findMaxima(spotsImage.getProcessor(),
                                       2,
                                       MaximumFinder.SINGLE_POINTS,
                                       true);
    ImagePlus maximaImp = new ImagePlus("A", byteProcessor);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(spotsImage).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.detectOptima(clij, src, dst, 1, true);
    ImagePlus maximaFromCL = clij.converter(dst).getImagePlus();
    //    maximaFromCL = new Duplicator().run(maximaFromCL, 2, 2);

    IJ.run(maximaImp, "Divide...", "value=255");

    assertTrue(TestUtilities.compareImages(maximaImp, maximaFromCL));

    src.close();
    dst.close();
  }

  @Test public void detectMaxima2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            spotsImage =
            NewImage.createImage("",
                    100,
                    100,
                    1,
                    16,
                    NewImage.FILL_BLACK);

    spotsImage.setZ(1);
    ImageProcessor ip1 = spotsImage.getProcessor();
    ip1.set(50, 50, 10);
    ip1.set(60, 60, 10);
    ip1.set(70, 70, 10);

    spotsImage.show();
    //IJ.run(spotsImage, "Find Maxima...", "noise=2 output=[Single Points]");

    ByteProcessor
            byteProcessor =
            new MaximumFinder().findMaxima(spotsImage.getProcessor(),
                    2,
                    MaximumFinder.SINGLE_POINTS,
                    true);
    ImagePlus maximaImp = new ImagePlus("A", byteProcessor);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(spotsImage).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.detectOptima(clij, src, dst, 1, true);
    ImagePlus maximaFromCL = clij.converter(dst).getImagePlus();
    //    maximaFromCL = new Duplicator().run(maximaFromCL, 2, 2);

    IJ.run(maximaImp, "Divide...", "value=255");

    assertTrue(TestUtilities.compareImages(maximaImp, maximaFromCL));

    src.close();
    dst.close();
  }

  @Test public void differenceOfGaussian3d()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for DoG");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.differenceOfGaussian(clij, src, dst, 6, 1.1f, 3.3f);

    src.close();
    dst.close();
  }

    @Test public void differenceOfGaussian3dSliceBySlice()
    {
        // do operation with ImageJ
        System.out.println("Todo: implement test for DoG slice by slice");

        // do operation with ClearCL
        ClearCLImage src = clij.converter(testImp1).getClearCLImage();
        ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

        Kernels.differenceOfGaussianSliceBySlice(clij, src, dst, 6, 1.1f, 3.3f);

        src.close();
        dst.close();
    }


    @Test public void differenceOfGaussian2d()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for DoG");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.differenceOfGaussian(clij, src, dst, 6, 1.1f, 3.3f);

    src.close();
    dst.close();
  }

  @Test public void dilate3d()
  {
    ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();
    ClearCLImage
            maskCLafter =
            clij.createCLImage(maskCL);

    Kernels.dilate(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 81);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void dilate3d_Buffers()
  {
    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter =
            clij.createCLBuffer(maskCL);

    Kernels.dilate(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 81);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void dilate2d()
  {
    ClearCLImage maskCL = clij.converter(mask2d).getClearCLImage();
    ClearCLImage
            maskCLafter = clij.createCLImage(maskCL);

    Kernels.dilate(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 21);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void dilate2d_Buffers()
  {
    ClearCLBuffer maskCL = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter = clij.createCLBuffer(maskCL);

    Kernels.dilate(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 21);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void dividePixelwise3d()
  {
    // do operation with ImageJ
    ImagePlus
            divided =
            new ImageCalculator().run("Divide create stack",
                    testImp1,
                    testImp2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.dividePixelwise(clij, src, src1, dst);

    ImagePlus dividedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(divided, dividedCL));

    src.close();
    src1.close();
    dst.close();
  }


  @Test public void dividePixelwise3d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            divided =
            new ImageCalculator().run("Divide create stack",
                    testImp1,
                    testImp2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.dividePixelwise(clij, src, src1, dst);

    ImagePlus dividedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(divided, dividedCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void dividePixelwise2d()
    {
        // do operation with ImageJ
        ImagePlus
                divided =
                new ImageCalculator().run("Divide create",
                        testImp2D1,
                        testImp2D2);

        // do operation with ClearCL
        ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
        ClearCLImage src1 = clij.converter(testImp2D2).getClearCLImage();
        ClearCLImage dst = clij.createCLImage(src);

        Kernels.dividePixelwise(clij, src, src1, dst);

        ImagePlus dividedCL = clij.converter(dst).getImagePlus();

        assertTrue(TestUtilities.compareImages(divided, dividedCL));

        src.close();
        src1.close();
        dst.close();
    }

  @Test public void dividePixelwise2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            divided =
            new ImageCalculator().run("Divide create",
                    testImp2D1,
                    testImp2D2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2D2).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.dividePixelwise(clij, src, src1, dst);

    ImagePlus dividedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(divided, dividedCL));

    src.close();
    src1.close();
    dst.close();
  }


  @Test public void downsample3d() throws InterruptedException {
    // do operation with ImageJ
    new ImageJ(); // the menu command 'Scale...' can only be executed successfully if the ImageJ UI is visible; apparently
    testImp1.show();
    IJ.run(testImp1, "Scale...", "x=0.5 y=0.5 z=0.5 width=512 height=1024 depth=5 interpolation=None process create");
    ImagePlus downsampled = IJ.getImage();



    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage dst =
            clij.createCLImage(new long[] { src.getWidth() / 2,
                            src.getHeight() / 2,
                            (long)(src.getDepth() - 0.5) / 2},
                    src.getChannelDataType());


    Kernels.downsample(clij, src, dst, 0.5f, 0.5f, 0.5f);

    ImagePlus downsampledCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(downsampled, downsampledCL, 1.0));
  }

  @Test public void downsample3d_Buffers() throws InterruptedException {
    // do operation with ImageJ
    new ImageJ(); // the menu command 'Scale...' can only be executed successfully if the ImageJ UI is visible; apparently
    testImp1.show();
    IJ.run(testImp1, "Scale...", "x=0.5 y=0.5 z=0.5 width=512 height=1024 depth=5 interpolation=None process create");
    ImagePlus downsampled = IJ.getImage();



    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer dst =
            clij.createCLBuffer(new long[] { src.getWidth() / 2,
                            src.getHeight() / 2,
                            (long)(src.getDepth() - 0.5) / 2},
                    src.getNativeType());


    Kernels.downsample(clij, src, dst, 0.5f, 0.5f, 0.5f);

    ImagePlus downsampledCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(downsampled, downsampledCL, 1.0));
  }

  @Test public void downsample2d() throws InterruptedException {
    // do operation with ImageJ
    new ImageJ(); // the menu command 'Scale...' can only be executed successfully if the ImageJ UI is visible; apparently
    testImp2D1.show();
    IJ.run(testImp2D1, "Scale...", "x=0.5 y=0.5 width=50 height=50 interpolation=None create");
    ImagePlus downsampled = IJ.getImage();

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(new long[] { src.getWidth() / 2,
                                        src.getHeight() / 2 },
                           src.getChannelDataType());

    Kernels.downsample(clij, src, dst, 0.5f, 0.5f);

    ImagePlus downsampledCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(downsampled, downsampledCL, 1.0));

  }


  @Test public void downsample2d_Buffers() throws InterruptedException {
    // do operation with ImageJ
    new ImageJ(); // the menu command 'Scale...' can only be executed successfully if the ImageJ UI is visible; apparently
    testImp2D1.show();
    IJ.run(testImp2D1, "Scale...", "x=0.5 y=0.5 width=50 height=50 interpolation=None create");
    ImagePlus downsampled = IJ.getImage();

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer
            dst =
            clij.createCLBuffer(new long[] { src.getWidth() / 2,
                            src.getHeight() / 2 },
                    src.getNativeType());

    Kernels.downsample(clij, src, dst, 0.5f, 0.5f);

    ImagePlus downsampledCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(downsampled, downsampledCL, 1.0));

  }

  @Test public void downsampleSliceBySliceMedian() {
    Img<ByteType> testImgBig = ArrayImgs.bytes(new byte[]{
            1, 2, 4, 4,
            2, 3, 4, 4,
            0, 1, 0, 0,
            1, 2, 0, 0
    }, new long[]{4,4,1});

    Img<ByteType> testImgSmall = ArrayImgs.bytes(new byte[]{
            2, 4,
            1, 0
    }, new long[]{2,2,1});

    ClearCLImage inputCL = clij.converter(testImgBig).getClearCLImage();
    ClearCLImage outputCL = clij.createCLImage(new long[]{2,2,1}, ImageChannelDataType.SignedInt8);

    Kernels.downsampleSliceBySliceHalfMedian(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();
    ImagePlus reference = clij.converter(testImgSmall).getImagePlus();

    assertTrue(TestUtilities.compareImages(result, reference));
  }

  @Test public void downsampleSliceBySliceMedian_Buffer() {
    Img<ByteType> testImgBig = ArrayImgs.bytes(new byte[]{
            1, 2, 4, 4,
            2, 3, 4, 4,
            0, 1, 0, 0,
            1, 2, 0, 0
    }, new long[]{4,4,1});

    Img<ByteType> testImgSmall = ArrayImgs.bytes(new byte[]{
            2, 4,
            1, 0
    }, new long[]{2,2,1});

    ClearCLBuffer inputCL = clij.converter(testImgBig).getClearCLBuffer();
    ClearCLBuffer outputCL = clij.createCLBuffer(new long[]{2,2,1}, NativeTypeEnum.Byte);

    Kernels.downsampleSliceBySliceHalfMedian(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();
    ImagePlus reference = clij.converter(testImgSmall).getImagePlus();

    assertTrue(TestUtilities.compareImages(result, reference));
  }

  @Test public void erode3d()
  {
    ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();
    ClearCLImage
        maskCLafter = clij.createCLImage(maskCL);

    Kernels.erode(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 1);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void erode3d_Buffer()
  {
    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter = clij.createCLBuffer(maskCL);

    Kernels.erode(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 1);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void erode2d()
  {
    ClearCLImage maskCL = clij.converter(mask2d).getClearCLImage();
    ClearCLImage
            maskCLafter = clij.createCLImage(maskCL);

    Kernels.erode(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 1);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void erode2d_Buffers()
  {
    ClearCLBuffer maskCL = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter = clij.createCLBuffer(maskCL);

    Kernels.erode(clij, maskCL, maskCLafter);

    double sum = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sum == 1);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void flip3d() throws InterruptedException
  {
    ClearCLImage testCL = clij.converter(testImp1).getClearCLImage();
    ClearCLImage flip = clij.converter(testImp1).getClearCLImage();
    ClearCLImage flop = clij.converter(testImp1).getClearCLImage();

    Kernels.flip(clij, testCL, flip, true, false, false);

    ImagePlus testFlipped = clij.converter(flip).getImagePlus();

    Kernels.flip(clij, flip, flop, true, false, false);
    ImagePlus testFlippedTwice = clij.converter(flop).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp1,
                                           testFlippedTwice));
    assertFalse(TestUtilities.compareImages(testImp1, testFlipped));

    testCL.close();
    flip.close();
    flop.close();
  }

  @Test public void flip2d()
  {
    ClearCLImage
        testCL =
        clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage flip = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage flop = clij.converter(testImp2D1).getClearCLImage();

    Kernels.flip(clij, testCL, flip, true, false);

    ImagePlus testFlipped = clij.converter(flip).getImagePlus();

    Kernels.flip(clij, flip, flop, true, false);
    ImagePlus testFlippedTwice = clij.converter(flop).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp2D1,
                                           testFlippedTwice));
    assertFalse(TestUtilities.compareImages(testImp2D1, testFlipped));

    testCL.close();
    flip.close();
    flop.close();
  }

  @Test public void flipBuffer3d()
  {
    ClearCLBuffer
        testCL =
        clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer flip = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer flop = clij.converter(testImp1).getClearCLBuffer();

    Kernels.flip(clij, testCL, flip, true, false, false);

    ImagePlus testFlipped = clij.converter(flip).getImagePlus();

    Kernels.flip(clij, flip, flop, true, false, false);
    ImagePlus testFlippedTwice = clij.converter(flop).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp1,
                                           testFlippedTwice));
    assertFalse(TestUtilities.compareImages(testImp1, testFlipped));

    testCL.close();
    flip.close();
    flop.close();
  }

  @Test public void flipBuffer2d()
  {
    ClearCLBuffer
        testCL =
        clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer
        flip =
        clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer
        flop =
        clij.converter(testImp2D1).getClearCLBuffer();

    Kernels.flip(clij, testCL, flip, true, false);

    ImagePlus testFlipped = clij.converter(flip).getImagePlus();

    Kernels.flip(clij, flip, flop, true, false);
    ImagePlus testFlippedTwice = clij.converter(flop).getImagePlus();

    assertTrue(TestUtilities.compareImages(testImp2D1,
                                           testFlippedTwice));
    assertFalse(TestUtilities.compareImages(testImp2D1, testFlipped));

    testCL.close();
    flip.close();
    flop.close();
  }

  @Test public void invertBinary3d()
  {
    ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();
    ClearCLImage
            maskCLafter = clij.createCLImage(maskCL);

    Kernels.invertBinary(clij, maskCL, maskCLafter);

    double sumCL = Kernels.sumPixels(clij, maskCL);
    double sumCLafter = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sumCLafter
            == maskCL.getWidth()
            * maskCL.getHeight()
            * maskCL.getDepth() - sumCL);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void invertBinary3d_Buffer()
  {
    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter = clij.createCLBuffer(maskCL);

    Kernels.invertBinary(clij, maskCL, maskCLafter);

    double sumCL = Kernels.sumPixels(clij, maskCL);
    double sumCLafter = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sumCLafter
            == maskCL.getWidth()
            * maskCL.getHeight()
            * maskCL.getDepth() - sumCL);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void invertBinary2d()
  {
    ClearCLImage maskCL = clij.converter(mask2d).getClearCLImage();
    ClearCLImage
            maskCLafter = clij.createCLImage(maskCL);

    Kernels.invertBinary(clij, maskCL, maskCLafter);

    double sumCL = Kernels.sumPixels(clij, maskCL);
    double sumCLafter = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sumCLafter
            == maskCL.getWidth()
            * maskCL.getHeight()
            * maskCL.getDepth() - sumCL);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void invertBinary2d_Buffer()
  {
    ClearCLBuffer maskCL = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer
            maskCLafter = clij.createCLBuffer(maskCL);

    Kernels.invertBinary(clij, maskCL, maskCLafter);

    double sumCL = Kernels.sumPixels(clij, maskCL);
    double sumCLafter = Kernels.sumPixels(clij, maskCLafter);

    assertTrue(sumCLafter
            == maskCL.getWidth()
            * maskCL.getHeight()
            * maskCL.getDepth() - sumCL);

    maskCL.close();
    maskCLafter.close();
  }

  @Test public void localThreshold3D() {
    ClearCLImage input = clij.converter(testImp1).getClearCLImage();
    ClearCLImage output1 = clij.createCLImage(input);
    ClearCLImage output2 = clij.createCLImage(input);
    ClearCLImage temp = clij.createCLImage(input);
    ClearCLImage blurred = clij.createCLImage(input);

    Kernels.blurSeparable(clij, input, blurred, new float[]{2, 2, 2});

    // usual way: blur, subtract, threshold
    ElapsedTime.measureForceOutput("traditional thresholding", () -> {
      Kernels.addWeightedPixelwise(clij, input, blurred, temp, 1f, -1f);
      Kernels.threshold(clij, temp, output1, 0f);
    });

    // short cut: blur, local threshold
    ElapsedTime.measureForceOutput("local threshold", () -> {
      Kernels.localThreshold(clij, input, output2, blurred);
    });

    System.out.println("O1: " + Kernels.sumPixels(clij, output1));
    System.out.println("O2: " + Kernels.sumPixels(clij, output2));

    assertTrue(Kernels.sumPixels(clij, output1) > 0);
    assertTrue(Kernels.sumPixels(clij, output1) == Kernels.sumPixels(clij, output2));

    Kernels.addWeightedPixelwise(clij, output1, output2, temp, 1f, -1f);

    assertTrue(Kernels.sumPixels(clij, temp) == 0);

    input.close();
    output1.close();
    output2.close();
    temp.close();
    blurred.close();

  }

  @Test public void mask3d()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for mask3d");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage mask = clij.converter(mask3d).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(mask);

    Kernels.mask(clij, src, mask, dst);

    src.close();
    mask.close();
    dst.close();
  }

  @Test public void mask3d_Buffers()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for mask3d");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer mask = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(mask);

    Kernels.mask(clij, src, mask, dst);

    src.close();
    mask.close();
    dst.close();
  }

  @Test public void mask2d()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for mask2d");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage mask = clij.converter(mask2d).getClearCLImage();
    ClearCLImage dst = clij.converter(mask2d).getClearCLImage();

    Kernels.mask(clij, src, mask, dst);

    mask.close();
    dst.close();
  }

  @Test public void mask2d_Buffers()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for mask2d");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer mask = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(mask2d).getClearCLBuffer();

    Kernels.mask(clij, src, mask, dst);

    mask.close();
    dst.close();
  }

  @Test public void maskStackWithPlane()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for maskStackWithPlane");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage mask = clij.converter(mask2d).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.maskStackWithPlane(clij, src, mask, dst);

    mask.close();
    dst.close();

  }

  @Test public void maskStackWithPlane_Buffers()
  {
    // do operation with ImageJ
    System.out.println("Todo: implement test for maskStackWithPlane");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer mask = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.maskStackWithPlane(clij, src, mask, dst);

    mask.close();
    dst.close();

  }

  @Test public void maximum2d() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Maximum...", "radius=1");

    // do operation with ClearCLIJ
    ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
    ClearCLImage outputCl = clij.createCLImage(inputCL);

    Kernels.maximum(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

  @Test public void maximum2d_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Maximum...", "radius=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.maximum(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference);
    result = new Duplicator().run(result);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

  @Test public void maximum3d() {
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Maximum 3D...", "x=1 y=1 z=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.maximum(clij, inputCL, outputCl, 3, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        // ignore edges and first and last slice
        reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
        result = new Duplicator().run(result, 2, result.getNSlices() - 2);

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void maximum3d_Buffers() {
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Maximum 3D...", "x=1 y=1 z=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.maximum(clij, inputCL, outputCl, 3, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges and first and last slice
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
    result = new Duplicator().run(result, 2, result.getNSlices() - 2);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void maximumSliceBySlice() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Maximum...", "radius=1 stack");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.maximumSliceBySlice(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void maximumSliceBySlice_Buffer() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Maximum...", "radius=1 stack");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.maximumSliceBySlice(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference);
    result = new Duplicator().run(result);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

  @Test public void maxPixelWise3d() {
    System.out.println("Todo: implement test for maxPixelwise3d");
  }

  @Test public void maxProjection() throws InterruptedException
  {
    // do operation with ImageJ
    ImagePlus
        maxProjection =
        NewImage.createShortImage("",
                                  testImp1.getWidth(),
                                  testImp2.getHeight(),
                                  1,
                                  NewImage.FILL_BLACK);
    ImageProcessor ipMax = maxProjection.getProcessor();

    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    for (int z = 0; z < testImp1copy.getNSlices(); z++)
    {
      testImp1copy.setZ(z + 1);
      ImageProcessor ip = testImp1copy.getProcessor();
      for (int x = 0; x < testImp1copy.getWidth(); x++)
      {
        for (int y = 0; y < testImp1copy.getHeight(); y++)
        {
          float value = ip.getf(x, y);
          if (value > ipMax.getf(x, y))
          {
            ipMax.setf(x, y, value);
          }
        }
      }
    }

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage
        dst =
        clij.createCLImage(new long[] { src.getWidth(),
                                        src.getHeight() },
                           src.getChannelDataType());

    Kernels.maxProjection(clij, src, dst);

    ImagePlus maxProjectionCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(maxProjection,
                                           maxProjectionCL));

    src.close();
    dst.close();
  }

  @Test public void maxProjection_Buffers() throws InterruptedException
  {
    // do operation with ImageJ
    ImagePlus
            maxProjection =
            NewImage.createShortImage("",
                    testImp1.getWidth(),
                    testImp2.getHeight(),
                    1,
                    NewImage.FILL_BLACK);
    ImageProcessor ipMax = maxProjection.getProcessor();

    ImagePlus testImp1copy = new Duplicator().run(testImp1);
    for (int z = 0; z < testImp1copy.getNSlices(); z++)
    {
      testImp1copy.setZ(z + 1);
      ImageProcessor ip = testImp1copy.getProcessor();
      for (int x = 0; x < testImp1copy.getWidth(); x++)
      {
        for (int y = 0; y < testImp1copy.getHeight(); y++)
        {
          float value = ip.getf(x, y);
          if (value > ipMax.getf(x, y))
          {
            ipMax.setf(x, y, value);
          }
        }
      }
    }

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer
            dst =
            clij.createCLBuffer(new long[] { src.getWidth(),
                            src.getHeight() },
                    src.getNativeType());

    Kernels.maxProjection(clij, src, dst);

    ImagePlus maxProjectionCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(maxProjection,
            maxProjectionCL));

    src.close();
    dst.close();
  }

  @Test public void mean2d() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Mean...", "radius=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.mean(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }


  @Test public void mean2d_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Mean...", "radius=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.mean(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference);
    result = new Duplicator().run(result);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void mean3d() {
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Mean 3D...", "x=1 y=1 z=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.mean(clij, inputCL, outputCl, 3, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        // ignore edges and first and last slice
        reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
        result = new Duplicator().run(result, 2, result.getNSlices() - 2);

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void mean3d_Buffers() {
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Mean 3D...", "x=1 y=1 z=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.mean(clij, inputCL, outputCl, 3, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges and first and last slice
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
    result = new Duplicator().run(result, 2, result.getNSlices() - 2);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

  @Test public void meanSliceBySlice() {

      ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
      IJ.run(testImage, "32-bit", "");

      // do operation with ImageJ
      ImagePlus reference = new Duplicator().run(testImage);
      IJ.run(reference, "Mean...", "radius=1 stack");

      // do operation with ClearCLIJ
      ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
      ClearCLImage outputCl = clij.createCLImage(inputCL);

      Kernels.meanSliceBySlice(clij, inputCL, outputCl, 3, 3);

      ImagePlus result = clij.converter(outputCl).getImagePlus();

      //new ImageJ();
      //clij.show(inputCL, "inp");
      //clij.show(reference, "ref");
      //clij.show(result, "res");
      //new WaitForUserDialog("wait").show();
      assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

  @Test public void meanSliceBySlice_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Mean...", "radius=1 stack");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.meanSliceBySlice(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges and first and last slice
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
    result = new Duplicator().run(result, 2, result.getNSlices() - 2);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void median2d() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Median...", "radius=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.median(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void median2d_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Median...", "radius=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.median(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void median3d() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Median 3D...", "x=1 y=1 z=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.median(clij, inputCL, outputCl, 3, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        // ignore edges and first and last slice
        reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
        result = new Duplicator().run(result, 2, result.getNSlices() - 2);

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void median3d_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Median 3D...", "x=1 y=1 z=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.median(clij, inputCL, outputCl, 3, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges and first and last slice
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
    result = new Duplicator().run(result, 2, result.getNSlices() - 2);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void medianSliceBySlice() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Median...", "radius=1 stack");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.medianSliceBySlice(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void medianSliceBySlice_Buffer() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Median...", "radius=1 stack");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.medianSliceBySlice(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference);
    result = new Duplicator().run(result);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void minimum2d() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Minimum...", "radius=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.minimum(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }


  @Test public void minimum2d_Buffers() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D, 20, 20);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Minimum...", "radius=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.minimum(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

   @Test public void minimum3d() {
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Minimum 3D...", "x=1 y=1 z=1");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.minimum(clij, inputCL, outputCl, 3, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        // ignore edges and first and last slice
        reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
        reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
        result = new Duplicator().run(result, 2, result.getNSlices() - 2);

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void minimum3d_Buffer() {
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Minimum 3D...", "x=1 y=1 z=1");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.minimum(clij, inputCL, outputCl, 3, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    // ignore edges and first and last slice
    reference.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    result.setRoi(new Roi(1,1, reference.getWidth() - 2, reference.getHeight() - 2));
    reference = new Duplicator().run(reference, 2, result.getNSlices() - 2);
    result = new Duplicator().run(result, 2, result.getNSlices() - 2);

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void minimumSliceBySlice() {

        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        IJ.run(testImage, "32-bit", "");

        // do operation with ImageJ
        ImagePlus reference = new Duplicator().run(testImage);
        IJ.run(reference, "Minimum...", "radius=1 stack");

        // do operation with ClearCLIJ
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCl = clij.createCLImage(inputCL);

        Kernels.minimumSliceBySlice(clij, inputCL, outputCl, 3, 3);

        ImagePlus result = clij.converter(outputCl).getImagePlus();

        //new ImageJ();
        //clij.show(inputCL, "inp");
        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();
        assertTrue(TestUtilities.compareImages(reference, result, 0.001));
    }

  @Test public void minimumSliceBySlice_Buffer() {

    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    IJ.run(testImage, "32-bit", "");

    // do operation with ImageJ
    ImagePlus reference = new Duplicator().run(testImage);
    IJ.run(reference, "Minimum...", "radius=1 stack");

    // do operation with ClearCLIJ
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCl = clij.createCLBuffer(inputCL);

    Kernels.minimumSliceBySlice(clij, inputCL, outputCl, 3, 3);

    ImagePlus result = clij.converter(outputCl).getImagePlus();

    //new ImageJ();
    //clij.show(inputCL, "inp");
    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();
    assertTrue(TestUtilities.compareImages(reference, result, 0.001));
  }

    @Test public void multiplyPixelwise3d()
  {
    // do operation with ImageJ
    ImagePlus
        multiplied =
        new ImageCalculator().run("Multiply create stack",
                                  testImp1,
                                  testImp2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

    Kernels.multiplyPixelwise(clij, src, src1, dst);

    ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Ignore
  @Test public void multiplyPixelwise3dathousandtimes() {
    // do operation with ImageJ
    ImagePlus
            multiplied =
            new ImageCalculator().run("Multiply create stack",
                    testImp1,
                    testImp2);

    for (int i = 0; i < 1000; i++) {
      // do operation with ClearCL
      ClearCLImage src = clij.converter(testImp1).getClearCLImage();
      ClearCLImage src1 = clij.converter(testImp2).getClearCLImage();
      ClearCLImage dst = clij.converter(testImp1).getClearCLImage();

      Kernels.multiplyPixelwise(clij, src, src1, dst);

      ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

      assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

      src.close();
      src1.close();
      dst.close();
    }
  }

  @Test public void multiplyPixelwise3d_Buffers() {
    // do operation with ImageJ
    ImagePlus
            multiplied =
            new ImageCalculator().run("Multiply create stack",
                    testImp1,
                    testImp2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp1).getClearCLBuffer();

    Kernels.multiplyPixelwise(clij, src, src1, dst);

    ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Ignore
  @Test public void multiplyPixelwise3d_Buffers_athousandtimes() {
    // do operation with ImageJ
    ImagePlus
            multiplied =
            new ImageCalculator().run("Multiply create stack",
                    testImp1,
                    testImp2);

    for (int i = 0; i < 1000; i++) {
      // do operation with ClearCL
      ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
      ClearCLBuffer src1 = clij.converter(testImp2).getClearCLBuffer();
      ClearCLBuffer dst = clij.converter(testImp1).getClearCLBuffer();

      Kernels.multiplyPixelwise(clij, src, src1, dst);

      ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

      assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

      src.close();
      src1.close();
      dst.close();
    }
  }


  @Test public void multiplyPixelwise2d()
  {
    // do operation with ImageJ
    ImagePlus
        multiplied =
        new ImageCalculator().run("Multiply create",
                                  testImp2D1,
                                  testImp2D2);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage src1 = clij.converter(testImp2D2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D1).getClearCLImage();

    Kernels.multiplyPixelwise(clij, src, src1, dst);

    ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void multiplyPixelwise2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus
            multiplied =
            new ImageCalculator().run("Multiply create",
                    testImp2D1,
                    testImp2D2);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer src1 = clij.converter(testImp2D2).getClearCLBuffer();
    ClearCLBuffer dst = clij.converter(testImp2D1).getClearCLBuffer();

    Kernels.multiplyPixelwise(clij, src, src1, dst);

    ImagePlus multipliedCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(multiplied, multipliedCL));

    src.close();
    src1.close();
    dst.close();
  }

  @Test public void multiplyScalar3d()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp1);
    IJ.run(added, "Multiply...", "value=2 stack");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.multiplyScalar(clij, src, dst, 2f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }


  @Test public void multiplyScalar3d_Buffer()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp1);
    IJ.run(added, "Multiply...", "value=2 stack");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.multiplyScalar(clij, src, dst, 2f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void multiplyScalar2d()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp2D1);
    IJ.run(added, "Multiply...", "value=2");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.multiplyScalar(clij, src, dst, 2f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void multiplyScalar2d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus added = new Duplicator().run(testImp2D1);
    IJ.run(added, "Multiply...", "value=2");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.multiplyScalar(clij, src, dst, 2f);
    ImagePlus addedFromCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(added, addedFromCL));

    src.close();
    dst.close();
  }

  @Test public void multiplySliceBySliceWithScalars() {

      ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();
      ClearCLImage multipliedBy2 = clij.createCLImage(maskCL);

      float[] factors = new float[(int)maskCL.getDepth()];
      for (int i = 0; i < factors.length; i++) {
        factors[i] = 2;
      }
      Kernels.multiplySliceBySliceWithScalars(clij, maskCL, multipliedBy2, factors);

      assertEquals(Kernels.sumPixels(clij, maskCL) * 2, Kernels.sumPixels(clij, multipliedBy2), 0.001);

      multipliedBy2.close();
      maskCL.close();

  }

  @Test public void multiplySliceBySliceWithScalars_Buffer() {

    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();
    ClearCLBuffer multipliedBy2 = clij.createCLBuffer(maskCL);

    float[] factors = new float[(int)maskCL.getDepth()];
    for (int i = 0; i < factors.length; i++) {
      factors[i] = 2;
    }
    Kernels.multiplySliceBySliceWithScalars(clij, maskCL, multipliedBy2, factors);

    assertEquals(Kernels.sumPixels(clij, maskCL) * 2, Kernels.sumPixels(clij, multipliedBy2), 0.001);

    multipliedBy2.close();
    maskCL.close();

  }

  @Test public void multiplyStackWithPlane()
  {
    // do operation with ImageJ
    System.out.println(
        "Todo: implement test for multiplyStackWithPlane");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp1).getClearCLImage();
    ClearCLImage mask = clij.converter(mask2d).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.multiplyStackWithPlane(clij, src, mask, dst);

    mask.close();
    dst.close();

  }

  @Test public void multiplyStackWithPlane_Buffers()
  {
    // do operation with ImageJ
    System.out.println(
            "Todo: implement test for multiplyStackWithPlane");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp1).getClearCLBuffer();
    ClearCLBuffer mask = clij.converter(mask2d).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.multiplyStackWithPlane(clij, src, mask, dst);

    mask.close();
    dst.close();

  }

  @Test public void power() {

    // do operation with ImageJ
    ImagePlus
            squared =
            new ImageCalculator().run("Multiply create",
                    testImp2D1,
                    testImp2D1);

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D1).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.power(clij, src, dst, 2.0f);

    ImagePlus squaredCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(squared, squaredCL));

    src.close();
    dst.close();


  }

  @Test public void power_Buffers() {

    // do operation with ImageJ
    ImagePlus
            squared =
            new ImageCalculator().run("Multiply create",
                    testImp2D1,
                    testImp2D1);

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2D1).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.power(clij, src, dst, 2.0f);

    ImagePlus squaredCL = clij.converter(dst).getImagePlus();

    assertTrue(TestUtilities.compareImages(squared, squaredCL));

    src.close();
    dst.close();


  }

    @Test public void resliceBottom() throws InterruptedException {

        testFlyBrain3D.setRoi(0,0, 256,128);
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        testImage.show();

        // do operation with ImageJ
        new ImageJ();
        IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Bottom avoid");
        Thread.sleep(500);
        ImagePlus reference = IJ.getImage();

        // do operation with OpenCL
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCL = clij.createCLImage(new long[] {inputCL.getWidth(), inputCL.getDepth(), inputCL.getHeight()}, inputCL.getChannelDataType());

        Kernels.resliceBottom(clij, inputCL, outputCL);

        ImagePlus result = clij.converter(outputCL).getImagePlus();

        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();

        assertTrue(TestUtilities.compareImages(reference, result));


    }

  @Test public void resliceBottom_Buffers() throws InterruptedException {

    testFlyBrain3D.setRoi(0,0, 256,128);
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    testImage.show();

    // do operation with ImageJ
    new ImageJ();
    IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Bottom avoid");
    Thread.sleep(500);
    ImagePlus reference = IJ.getImage();

    // do operation with OpenCL
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCL = clij.createCLBuffer(new long[] {inputCL.getWidth(), inputCL.getDepth(), inputCL.getHeight()}, inputCL.getNativeType());

    Kernels.resliceBottom(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();

    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();

    assertTrue(TestUtilities.compareImages(reference, result));


  }

  @Test public void resliceLeft() throws InterruptedException {

      testFlyBrain3D.setRoi(0,0, 256,128);
      ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
      testImage.show();

      // do operation with ImageJ
      new ImageJ();
      IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Left avoid");
      Thread.sleep(500);
      ImagePlus reference = IJ.getImage();

      // do operation with OpenCL
      ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
      ClearCLImage outputCL = clij.createCLImage(new long[] {inputCL.getHeight(), inputCL.getDepth(), inputCL.getWidth()}, inputCL.getChannelDataType());

      Kernels.resliceLeft(clij, inputCL, outputCL);

      ImagePlus result = clij.converter(outputCL).getImagePlus();

      //clij.show(reference, "ref");
      //clij.show(result, "res");
      //new WaitForUserDialog("wait").show();

      assertTrue(TestUtilities.compareImages(reference, result));


  }

  @Test public void resliceLeft_Buffers() throws InterruptedException {

    testFlyBrain3D.setRoi(0,0, 256,128);
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    testImage.show();

    // do operation with ImageJ
    new ImageJ();
    IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Left avoid");
    Thread.sleep(500);
    ImagePlus reference = IJ.getImage();

    // do operation with OpenCL
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCL = clij.createCLBuffer(new long[] {inputCL.getHeight(), inputCL.getDepth(), inputCL.getWidth()}, inputCL.getNativeType());

    Kernels.resliceLeft(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();

    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();

    assertTrue(TestUtilities.compareImages(reference, result));


  }

    @Test public void resliceRight() throws InterruptedException {

        testFlyBrain3D.setRoi(0,0, 256,128);
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        testImage.show();

        // do operation with ImageJ
        new ImageJ();
        IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Right avoid");
        Thread.sleep(500);
        ImagePlus reference = IJ.getImage();

        // do operation with OpenCL
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCL = clij.createCLImage(new long[] {inputCL.getHeight(), inputCL.getDepth(), inputCL.getWidth()}, inputCL.getChannelDataType());

        Kernels.resliceRight(clij, inputCL, outputCL);

        ImagePlus result = clij.converter(outputCL).getImagePlus();

        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();

        assertTrue(TestUtilities.compareImages(reference, result));


    }

  @Test public void resliceRight_Buffers() throws InterruptedException {

    testFlyBrain3D.setRoi(0,0, 256,128);
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    testImage.show();

    // do operation with ImageJ
    new ImageJ();
    IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Right avoid");
    Thread.sleep(500);
    ImagePlus reference = IJ.getImage();

    // do operation with OpenCL
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCL = clij.createCLBuffer(new long[] {inputCL.getHeight(), inputCL.getDepth(), inputCL.getWidth()}, inputCL.getNativeType());

    Kernels.resliceRight(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();

    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();

    assertTrue(TestUtilities.compareImages(reference, result));


  }

    @Test public void resliceTop() throws InterruptedException {

        testFlyBrain3D.setRoi(0,0, 256,128);
        ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
        testImage.show();

        // do operation with ImageJ
        new ImageJ();
        IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Top avoid");
        Thread.sleep(500);
        ImagePlus reference = IJ.getImage();

        // do operation with OpenCL
        ClearCLImage inputCL = clij.converter(testImage).getClearCLImage();
        ClearCLImage outputCL = clij.createCLImage(new long[] {inputCL.getWidth(), inputCL.getDepth(), inputCL.getHeight()}, inputCL.getChannelDataType());

        Kernels.resliceTop(clij, inputCL, outputCL);

        ImagePlus result = clij.converter(outputCL).getImagePlus();

        //clij.show(reference, "ref");
        //clij.show(result, "res");
        //new WaitForUserDialog("wait").show();

        assertTrue(TestUtilities.compareImages(reference, result));


    }

  @Test public void resliceTop_Buffers() throws InterruptedException {

    testFlyBrain3D.setRoi(0,0, 256,128);
    ImagePlus testImage = new Duplicator().run(testFlyBrain3D);
    testImage.show();

    // do operation with ImageJ
    new ImageJ();
    IJ.run(testImage, "Reslice [/]...", "output=1.0 start=Top avoid");
    Thread.sleep(500);
    ImagePlus reference = IJ.getImage();

    // do operation with OpenCL
    ClearCLBuffer inputCL = clij.converter(testImage).getClearCLBuffer();
    ClearCLBuffer outputCL = clij.createCLBuffer(new long[] {inputCL.getWidth(), inputCL.getDepth(), inputCL.getHeight()}, inputCL.getNativeType());

    Kernels.resliceTop(clij, inputCL, outputCL);

    ImagePlus result = clij.converter(outputCL).getImagePlus();

    //clij.show(reference, "ref");
    //clij.show(result, "res");
    //new WaitForUserDialog("wait").show();

    assertTrue(TestUtilities.compareImages(reference, result));


  }

  @Test public void set3d()
  {
    ClearCLImage imageCL = clij.converter(mask3d).getClearCLImage();

    Kernels.set(clij, imageCL, 2f);

    double sum = Kernels.sumPixels(clij, imageCL);

    assertTrue(sum
               == imageCL.getWidth()
                  * imageCL.getHeight()
                  * imageCL.getDepth()
                  * 2);

    imageCL.close();
  }

  @Test public void set3d_Buffers()
  {
    ClearCLBuffer imageCL = clij.converter(mask3d).getClearCLBuffer();

    Kernels.set(clij, imageCL, 2f);

    double sum = Kernels.sumPixels(clij, imageCL);

    assertTrue(sum
            == imageCL.getWidth()
            * imageCL.getHeight()
            * imageCL.getDepth()
            * 2);

    imageCL.close();
  }

  @Test public void set2d()
  {
    ClearCLImage imageCL = clij.converter(mask2d).getClearCLImage();

    Kernels.set(clij, imageCL, 2f);

    double sum = Kernels.sumPixels(clij, imageCL);

    assertTrue(sum == imageCL.getWidth() * imageCL.getHeight() * 2);

    imageCL.close();
  }


  @Test public void set2d_Buffers()
  {
    ClearCLBuffer imageCL = clij.converter(mask2d).getClearCLBuffer();

    Kernels.set(clij, imageCL, 2f);

    double sum = Kernels.sumPixels(clij, imageCL);

    assertTrue(sum == imageCL.getWidth() * imageCL.getHeight() * 2);

    imageCL.close();
  }

  @Test public void splitStack() {
    ClearCLImage clearCLImage = clij.converter(testFlyBrain3D).getClearCLImage();
    ClearCLImage split1 = clij.createCLImage(new long[]{clearCLImage.getWidth(), clearCLImage.getHeight(), clearCLImage.getDepth() / 2}, clearCLImage.getChannelDataType());
    ClearCLImage split2 = clij.createCLImage(new long[]{clearCLImage.getWidth(), clearCLImage.getHeight(), clearCLImage.getDepth() / 2}, clearCLImage.getChannelDataType());

    Kernels.splitStack(clij, clearCLImage, split1, split2);

    assertTrue(Kernels.sumPixels(clij, split1) > 0);
    assertTrue(Kernels.sumPixels(clij, split2) > 0);
  }

  @Test public void splitStack_Buffers() {
    ClearCLBuffer clearCLImage = clij.converter(testFlyBrain3D).getClearCLBuffer();
    ClearCLBuffer split1 = clij.createCLBuffer(new long[]{clearCLImage.getWidth(), clearCLImage.getHeight(), clearCLImage.getDepth() / 2}, clearCLImage.getNativeType());
    ClearCLBuffer split2 = clij.createCLBuffer(new long[]{clearCLImage.getWidth(), clearCLImage.getHeight(), clearCLImage.getDepth() / 2}, clearCLImage.getNativeType());

    Kernels.splitStack(clij, clearCLImage, split1, split2);

    assertTrue(Kernels.sumPixels(clij, split1) > 0);
    assertTrue(Kernels.sumPixels(clij, split2) > 0);
  }

  @Test public void sumPixelsSliceBySlice() {
    ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();

    double sum = Kernels.sumPixels(clij, maskCL);
    double[] sumSliceWise = Kernels.sumPixelsSliceBySlice(clij, maskCL);

    assertTrue(sum == new Sum().evaluate(sumSliceWise));

    maskCL.close();
  }

  @Test public void sumPixelsSliceBySlice_Buffers() {
    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();

    double sum = Kernels.sumPixels(clij, maskCL);
    double[] sumSliceWise = Kernels.sumPixelsSliceBySlice(clij, maskCL);

    assertTrue(sum == new Sum().evaluate(sumSliceWise));

    maskCL.close();
  }

  @Test public void sumPixels3d()
  {
    ClearCLImage maskCL = clij.converter(mask3d).getClearCLImage();

    double sum = Kernels.sumPixels(clij, maskCL);

    assertTrue(sum == 27);

    maskCL.close();
  }

  @Test public void sumPixels3d_Buffers()
  {
    ClearCLBuffer maskCL = clij.converter(mask3d).getClearCLBuffer();

    double sum = Kernels.sumPixels(clij, maskCL);

    assertTrue(sum == 27);

    maskCL.close();
  }

  @Test public void sumPixels2d()
  {
    ClearCLImage maskCL = clij.converter(mask2d).getClearCLImage();

    double sum = Kernels.sumPixels(clij, maskCL);

    assertTrue(sum == 9);

    maskCL.close();
  }

  @Test public void sumPixels2d_Buffers()
  {
    ClearCLBuffer maskCL = clij.converter(mask2d).getClearCLBuffer();

    double sum = Kernels.sumPixels(clij, maskCL);

    assertTrue(sum == 9);

    maskCL.close();
  }

  @Test public void tenengradWeights() {
    System.out.println("Todo: implement test for Tenengrad weights");
  }

  @Test public void tenengradFusion() {
      System.out.println("Todo: implement test for Tenengrad fusion");
  }

  @Test public void threshold3d()
  {
    // do operation with ImageJ
    ImagePlus thresholded = new Duplicator().run(testImp2);
    Prefs.blackBackground = false;
    IJ.setRawThreshold(thresholded, 2, 65535, null);
    IJ.run(thresholded,
            "Convert to Mask",
            "method=Default background=Dark");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2).getClearCLImage();
    ClearCLImage dst = clij.createCLImage(src);

    Kernels.threshold(clij, src, dst, 2f);
    Kernels.multiplyScalar(clij, dst, src, 255f);

    ImagePlus thresholdedCL = clij.converter(src).getImagePlus();

    assertTrue(TestUtilities.compareImages(thresholded,
            thresholdedCL));

    src.close();
    dst.close();
  }

  @Test public void threshold3d_Buffers()
  {
    // do operation with ImageJ
    ImagePlus thresholded = new Duplicator().run(testImp2);
    Prefs.blackBackground = false;
    IJ.setRawThreshold(thresholded, 2, 65535, null);
    IJ.run(thresholded,
            "Convert to Mask",
            "method=Default background=Dark");

    // do operation with ClearCL
    ClearCLBuffer src = clij.converter(testImp2).getClearCLBuffer();
    ClearCLBuffer dst = clij.createCLBuffer(src);

    Kernels.threshold(clij, src, dst, 2f);
    Kernels.multiplyScalar(clij, dst, src, 255f);

    ImagePlus thresholdedCL = clij.converter(src).getImagePlus();

    assertTrue(TestUtilities.compareImages(thresholded,
            thresholdedCL));

    src.close();
    dst.close();
  }

  @Test public void threshold2d()
  {
    // do operation with ImageJ
    ImagePlus thresholded = new Duplicator().run(testImp2D2);
    Prefs.blackBackground = false;
    IJ.setRawThreshold(thresholded, 2, 65535, null);
    IJ.run(thresholded,
           "Convert to Mask",
           "method=Default background=Dark");

    // do operation with ClearCL
    ClearCLImage src = clij.converter(testImp2D2).getClearCLImage();
    ClearCLImage dst = clij.converter(testImp2D2).getClearCLImage();

    Kernels.threshold(clij, src, dst, 2f);
    Kernels.multiplyScalar(clij, dst, src, 255f);

    ImagePlus thresholdedCL = clij.converter(src).getImagePlus();

    assertTrue(TestUtilities.compareImages(thresholded,
                                           thresholdedCL));

    src.close();
    dst.close();
  }
}