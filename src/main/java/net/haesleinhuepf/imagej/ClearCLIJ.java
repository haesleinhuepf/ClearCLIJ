package net.haesleinhuepf.imagej;

import clearcl.*;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.backend.jocl.ClearCLBackendJOCL;
import clearcl.enums.*;
import clearcl.util.ElapsedTime;
import coremem.enums.NativeTypeEnum;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Duplicator;
import net.haesleinhuepf.imagej.converters.CLIJConverterPlugin;
import net.haesleinhuepf.imagej.converters.CLIJConverterService;
import net.haesleinhuepf.imagej.utilities.CLInfo;
import net.haesleinhuepf.imagej.utilities.CLKernelExecutor;
import net.imagej.ImageJ;
import net.imagej.patcher.LegacyInjector;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import org.scijava.Context;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * ClearCLIJ is an entry point for ImageJ/OpenCL compatibility.
 * Simply create an instance using the SingleTon implementation:
 * <p>
 * clij = ClearCLIJ.getInstance();
 * <p>
 * Alternatively, you can get an instance associated to a particular
 * OpenCL device by handing over its name to the constructor
 * <p>
 * clji = new ClearCLIJ("geforce");
 * <p>
 * To get a list of available devices, call
 * ClearCLIJ.getAvailableDevices()  to learn more about these devices,
 * call ClearCLIJ.clinfo();
 * <p>
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * February 2018
 */
public class ClearCLIJ {

    private static ClearCLIJ sInstance = null;
    protected ClearCLContext mClearCLContext;
    private ClearCLDevice mClearCLDevice;
    private ClearCL mClearCL;
    private CLKernelExecutor
            mCLKernelExecutor = null;


    @Deprecated
    public ClearCLIJ(int deviceIndex) {

        ClearCLBackendInterface
                lClearCLBackend = new ClearCLBackendJOCL();

        mClearCL = new ClearCL(lClearCLBackend);

        ArrayList<ClearCLDevice> allDevices = mClearCL.getAllDevices();
        for (int i = 0; i < allDevices.size(); i++) {
            System.out.println(allDevices.get(i).getName());
        }

        mClearCLDevice = allDevices.get(deviceIndex);
        System.out.println("Using OpenCL device: " + mClearCLDevice.getName());


        mClearCLContext = mClearCLDevice.createContext();
    }

    /**
     * Deprecated: use getInstance(String) instead
     *
     * @param pDeviceNameMustContain device name
     */
    @Deprecated
    public ClearCLIJ(String pDeviceNameMustContain) {
        ClearCLBackendInterface
                lClearCLBackend = new ClearCLBackendJOCL();

        mClearCL = new ClearCL(lClearCLBackend);
        if (pDeviceNameMustContain == null || pDeviceNameMustContain.length() == 0) {
            mClearCLDevice = null;
        } else {
            mClearCLDevice = mClearCL.getDeviceByName(pDeviceNameMustContain);
        }

        if (mClearCLDevice == null) {
            System.out.println("No GPU name specified. Using first GPU device found.");
            for (ClearCLDevice device : mClearCL.getAllDevices()) {
                if (!device.getName().contains("CPU")) {
                    mClearCLDevice = device;
                }
            }
        }
        if (mClearCLDevice == null) {
            System.out.println("Warning: GPU device determination failed. Retrying using first device found.");
            mClearCLDevice = mClearCL.getAllDevices().get(0);
        }
        System.out.println("Using OpenCL device: " + mClearCLDevice.getName());

        mClearCLContext = mClearCLDevice.createContext();
    }

    public static ClearCLIJ getInstance() {
        return getInstance(null);
    }

    public static ClearCLIJ getInstance(String pDeviceNameMustContain) {
        if (sInstance == null) {
            sInstance = new ClearCLIJ(pDeviceNameMustContain);
        }
        return sInstance;
    }

    public static String clinfo() {
        return CLInfo.clinfo();
    }

    public static ArrayList<String> getAvailableDeviceNames() {
        ArrayList<String> lResultList = new ArrayList<>();

        ClearCLBackendInterface lClearCLBackend = ClearCLBackends.getBestBackend();
        ClearCL lClearCL = new ClearCL(lClearCLBackend);
        for (ClearCLDevice lDevice : lClearCL.getAllDevices()) {
            lResultList.add(lDevice.getName());
        }
        return lResultList;
    }

    public boolean execute(String pProgramFilename,
                           String pKernelname,
                           Map<String, Object> pParameterMap) {
        return execute(Object.class, pProgramFilename, pKernelname, pParameterMap);
    }

    public boolean execute(Class pAnchorClass,
                           String pProgramFilename,
                           String pKernelname,
                           Map<String, Object> pParameterMap) {
        return execute(pAnchorClass, pProgramFilename, pKernelname, null, pParameterMap);
    }

    public boolean execute(Class pAnchorClass,
                           String pProgramFilename,
                           String pKernelname,
                           long[] pGlobalsizes,
                           Map<String, Object> pParameterMap) {
        final boolean[] result = new boolean[1];

        for (String key : pParameterMap.keySet()) {
            System.out.println(key + " = " + pParameterMap.get(key));
        }

        ElapsedTime.measure("kernel + build " + pKernelname, () -> {
            if (mCLKernelExecutor == null) {
                try {
                    mCLKernelExecutor = new CLKernelExecutor(mClearCLContext,
                            pAnchorClass,
                            pProgramFilename,
                            pKernelname,
                            pGlobalsizes);
                } catch (IOException e) {
                    e.printStackTrace();
                    result[0] = false;
                    return;
                }
            } else {
                mCLKernelExecutor.setProgramFilename(pProgramFilename);
                mCLKernelExecutor.setKernelName(pKernelname);
                mCLKernelExecutor.setAnchorClass(pAnchorClass);
                mCLKernelExecutor.setParameterMap(pParameterMap);
                mCLKernelExecutor.setGlobalSizes(pGlobalsizes);
            }
            mCLKernelExecutor.setParameterMap(pParameterMap);
            result[0] = mCLKernelExecutor.enqueue(true);
        });
        return result[0];
    }

    public void dispose() {
        mClearCLContext.close();
        converterService = null;
        if (sInstance == this) {
            sInstance = null;
        }
    }

    /**
     * Deprecated because it should not be neccessary to get the context.
     * The context should be shadowed inside.
     *
     * @return
     */
    @Deprecated
    public ClearCLContext getClearCLContext() {
        return mClearCLContext;
    }

    public Map<String, Object> parameters(Object... pParameterList) {
        Map<String, Object> lResultMap = new HashMap<String, Object>();
        for (int i = 0; i < pParameterList.length; i += 2) {
            lResultMap.put((String) pParameterList[i], pParameterList[i + 1]);
        }
        return lResultMap;
    }

    public ClearCLImage createCLImage(ClearCLImage pInputImage) {
        return mClearCLContext.createImage(pInputImage);
    }

    public ClearCLImage createCLImage(long[] dimensions, ImageChannelDataType pImageChannelType) {

        return mClearCLContext.createImage(HostAccessType.ReadWrite,
                KernelAccessType.ReadWrite,
                ImageChannelOrder.R,
                pImageChannelType,
                dimensions);
    }

    public ClearCLBuffer createCLBuffer(ClearCLBuffer inputCL) {
        return createCLBuffer(inputCL.getDimensions(), inputCL.getNativeType());
    }

    public ClearCLBuffer createCLBuffer(long[] dimensions, NativeTypeEnum pNativeType) {
        return mClearCLContext.createBuffer(
                MemAllocMode.Best,
                HostAccessType.ReadWrite,
                KernelAccessType.ReadWrite,
                1L,
                pNativeType,
                dimensions
        );
    }


    public void show(RandomAccessibleInterval input, String title) {
        show(convert(input, ImagePlus.class), title);
    }

    public void show(ClearCLImage input, String title) {
        show(convert(input, ImagePlus.class), title);
    }

    public void show(ClearCLBuffer input, String title) {
        show(convert(input, ImagePlus.class), title);
    }

    public void show(ImagePlus input, String title) {
        ImagePlus imp = new Duplicator().run(input);
        imp.setTitle(title);
        imp.setZ(imp.getNSlices() / 2);
        imp.setC(imp.getNChannels() / 2);
        IJ.run(imp, "Enhance Contrast", "saturated=0.35");
        if (imp.getNChannels() > 1 && imp.getNSlices() == 1) {
            IJ.run(imp, "Properties...", "channels=1 slices=" + imp.getNChannels() + " frames=1 unit=pixel pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
        }
        imp.show();
    }

    public boolean close() {
        mClearCLContext.close();
        mClearCLContext.getDevice().close();
        if (sInstance == this) {
            sInstance = null;
        }
        return true;
    }

    private CLIJConverterService converterService = null;
    public void setConverterService(CLIJConverterService converterService) {
        this.converterService = converterService;
    }

    public <S, T> T convert(S source, Class<T> targetClass) {
        if (targetClass.isAssignableFrom(source.getClass())) {
            return (T) source;
        }
        if (converterService == null) {
            converterService = new Context(CLIJConverterService.class).service(CLIJConverterService.class);
                    //new ImageJ().getContext().service(CLIJConverterService.class);
        }
        CLIJConverterPlugin<S, T> converter = (CLIJConverterPlugin<S, T>) converterService.getConverter(source.getClass(), targetClass);
        return converter.convert(source);
    }
}
