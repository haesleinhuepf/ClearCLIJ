package net.haesleinhuepf.imagej.macro.modules;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import net.haesleinhuepf.imagej.kernels.Kernels;
import net.haesleinhuepf.imagej.macro.AbstractCLIJPlugin;
import net.haesleinhuepf.imagej.macro.CLIJMacroPlugin;
import net.haesleinhuepf.imagej.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.imagej.macro.documentation.OffersDocumentation;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 * 12 2018
 */
@Plugin(type = CLIJMacroPlugin.class, name = "CLIJ_blur2DFast")
public class Blur2DFast extends AbstractCLIJPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        float sigmaX = asFloat(args[2]);
        float sigmaY = asFloat(args[3]);

        if (containsCLBufferArguments()) {
            // convert all arguments to CLImages
            Object[] args = openCLImageArgs();
            boolean result = Kernels.blurSeparable(clij, (ClearCLImage) (args[0]), (ClearCLImage) (args[1]), sigmaX, sigmaY, 0);
            // copy result back to the bufffer
            Kernels.copy(clij, (ClearCLImage)args[1], (ClearCLBuffer)this.args[1]);
            // cleanup
            releaseImages(args);
            return result;
        } else {
            return Kernels.blurSeparable(clij, (ClearCLImage)( args[0]), (ClearCLImage)(args[1]), sigmaX, sigmaY, 0);
        }
    }


    @Override
    public String getParameterHelpText() {
        return "Image source, Image destination, Number sigmaX, Number sigmaY";
    }

    @Override
    public String getDescription() {
        return "Computes the Gaussian blurred image of an image given two sigma values in X and Y. Thus, the filter" +
                "kernel can have non-isotropic shape.\n\n" +
                "" +
                "The 'fast' implementation is done separable. In case a sigma equals zero, the direction is not blurred.";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }

}
