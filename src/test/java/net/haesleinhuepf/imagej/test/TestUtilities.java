package net.haesleinhuepf.imagej.test;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.type.numeric.RealType;

/**
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * February 2018
 */
public class TestUtilities
{

  public static <T extends RealType<T>> boolean compareIterableIntervals(
      IterableInterval<T> lIterableInterval1,
      IterableInterval<T> lIterableInterval2)
  {
    double lSum = 0;
    Cursor<T> lCursor1 = lIterableInterval1.cursor();
    Cursor<T> lCursor2 = lIterableInterval2.cursor();

    while (lCursor1.hasNext() && lCursor2.hasNext())
    {
      if (lCursor1.next().getRealFloat() != lCursor2.next()
                                                    .getRealFloat())
      {
        System.out.println("lCursor1 " + lCursor1.get()
                                                 .getRealFloat());
        System.out.println("lCursor2 " + lCursor2.get()
                                                 .getRealFloat());
        System.out.println("Value different ");
        return false;
      }
      lSum += lCursor1.get().getRealDouble();
    }
    System.out.println("sum " + lSum);

    // check if one image is longer than the other
    if (lCursor1.hasNext() || lCursor2.hasNext())
    {
      return false;
    }

    return true;
  }

  public static boolean compareImages(ImagePlus a, ImagePlus b) {
    return compareImages(a, b, 0);
  }

  public static boolean compareImages(ImagePlus a, ImagePlus b, double tolerance)
  {
    if (a.getWidth() != b.getWidth()
        || a.getHeight() != b.getHeight()
        || a.getNChannels() != b.getNChannels()
        || a.getNFrames() != b.getNFrames()
        || a.getNSlices() != b.getNSlices())
    {
      System.out.println("sizes different");
      System.out.println("w " + a.getWidth() + " != " + b.getWidth());
      System.out.println("h "
                         + a.getHeight()
                         + " != "
                         + b.getHeight());
      System.out.println("c "
                         + a.getNChannels()
                         + " != "
                         + b.getNChannels());
      System.out.println("f "
                         + a.getNFrames()
                         + " != "
                         + b.getNFrames());
      System.out.println("s "
                         + a.getNSlices()
                         + " != "
                         + b.getNSlices());
      return false;
    }

    for (int c = 0; c < a.getNChannels(); c++)
    {
      a.setC(c + 1);
      b.setC(c + 1);
      for (int t = 0; t < a.getNFrames(); t++)
      {
        a.setT(t + 1);
        b.setT(t + 1);
        for (int z = 0; z < a.getNSlices(); z++)
        {
          a.setZ(z + 1);
          b.setZ(z + 1);
          ImageProcessor aIP = a.getProcessor();
          ImageProcessor bIP = b.getProcessor();
          for (int x = 0; x < a.getWidth(); x++)
          {
            for (int y = 0; y < a.getHeight(); y++)
            {
              if (Math.abs(aIP.getPixelValue(x, y) - bIP.getPixelValue(x, y)) > tolerance)
              {
                System.out.println("pixels different "
                                   + "| " + aIP.getPixelValue(x, y)
                                   + " - "
                                   + bIP.getPixelValue(x, y)
                                   + " | > " + tolerance);
                return false;
              }
            }
          }
        }
      }
    }
    return true;
  }

  public static boolean compareArrays(int[][][] a, int[][][] b) {
    if (a.length != b.length) {
      System.out.println("Array sizes differ");
      return false;
    }
    for (int x = 0; x < a.length; x++) {
      if (a[x].length != b[x].length) {
        System.out.println("Array[" + x + "] sizes differ");
        return false;
      }
      for (int y = 0; y < a[0].length; y++) {
        if (a[x][y].length != b[x][y].length) {
          System.out.println("Array[" + x + "][" + y + "] sizes differ");
          return false;
        }
        for (int z = 0; z < a[0][0].length; z++) {
          if(a[x][y][z] != b[x][y][z]) {
            System.out.println("Pixels[" + x + "][" + y + "][" + z + "] differ: " +
                    a[x][y][z] +
                    " != " +
                    b[x][y][z]
            );
            return false;
          }
        }
      }
    }
    return true;
  }

}
