__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void binary_or_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_IN_2D  src2,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = READ_IMAGE_2D(src1, sampler, pos).x;
  DTYPE_OUT value2 = READ_IMAGE_2D(src2, sampler, pos).x;
  if ( value1 > 0 || value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_and_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_IN_2D  src2,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = READ_IMAGE_2D(src1, sampler, pos).x;
  DTYPE_OUT value2 = READ_IMAGE_2D(src2, sampler, pos).x;
  if ( value1 > 0 && value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_not_2d(DTYPE_IMAGE_IN_2D  src1,
                           DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value1 = READ_IMAGE_2D(src1, sampler, pos).x;
  if ( value1 > 0) {
    value1 = 0;
  } else {
    value1 = 1;
  }
  WRITE_IMAGE_2D (dst, pos, value1);
}

__kernel void binary_or_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_IN_3D  src2,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = READ_IMAGE_3D(src1, sampler, pos).x;
  DTYPE_OUT value2 = READ_IMAGE_3D(src2, sampler, pos).x;
  if ( value1 > 0 || value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void binary_and_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_IN_3D  src2,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = READ_IMAGE_3D(src1, sampler, pos).x;
  DTYPE_OUT value2 = READ_IMAGE_3D(src2, sampler, pos).x;
  if ( value1 > 0 && value2 > 0 ) {
    value1 = 1;
  } else {
    value1 = 0;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}

__kernel void binary_not_3d(DTYPE_IMAGE_IN_3D  src1,
                           DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value1 = READ_IMAGE_3D(src1, sampler, pos).x;
  if ( value1 > 0) {
    value1 = 0;
  } else {
    value1 = 1;
  }
  WRITE_IMAGE_3D (dst, pos, value1);
}


__kernel void erode_diamond_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value > 0) {
    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value > 0) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value > 0) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value > 0) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
          if (value > 0) {
            value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, 1, 0})).x;
            if (value > 0) {
              value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, -1, 0})).x;
            }
          }
        }
      }
    }
  }

  WRITE_IMAGE_3D (dst, pos, value);
}


__kernel void erode_diamond_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value > 0) {
    value = READ_IMAGE_2D(src, sampler, (pos + (int2){1, 0})).x;
    if (value > 0) {
      value = READ_IMAGE_2D(src, sampler, (pos + (int2){-1, 0})).x;
      if (value > 0) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, 1})).x;
        if (value > 0) {
          value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, -1})).x;
        }
      }
    }
  }

  WRITE_IMAGE_2D (dst, pos, value);
}

__kernel void dilate_diamond_neighborhood_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value < 1) {

    value = READ_IMAGE_3D(src, sampler, (pos + (int4){1, 0, 0, 0})).x;
    if (value < 1) {
      value = READ_IMAGE_3D(src, sampler, (pos + (int4){-1, 0, 0, 0})).x;
      if (value < 1) {
        value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 1, 0, 0})).x;
        if (value < 1) {
          value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, -1, 0, 0})).x;
          if (value < 1) {
            value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, 1, 0})).x;
            if (value < 1) {
              value = READ_IMAGE_3D(src, sampler, (pos + (int4){0, 0, -1, 0})).x;
            }
          }
        }
      }
    }
  }

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void dilate_diamond_neighborhood_2d(DTYPE_IMAGE_IN_2D  src,
                          DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value < 1) {
    value = READ_IMAGE_2D(src, sampler, (pos + (int2){1, 0})).x;
    if (value < 1) {
      value = READ_IMAGE_2D(src, sampler, (pos + (int2){-1, 0})).x;
      if (value < 1) {
        value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, 1})).x;
        if (value < 1) {
          value = READ_IMAGE_2D(src, sampler, (pos + (int2){0, -1})).x;
        }
      }
    }
  }

  WRITE_IMAGE_2D (dst, pos, value);
}

__kernel void invert_3d(DTYPE_IMAGE_IN_3D  src,
                        DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  DTYPE_OUT value = READ_IMAGE_3D(src, sampler, pos).x;
  if (value != 0) {
    value = 0;
  } else {
    value = 1;
  }

  WRITE_IMAGE_3D (dst, pos, value);
}

__kernel void invert_2d(DTYPE_IMAGE_IN_2D  src,
                        DTYPE_IMAGE_OUT_2D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  DTYPE_OUT value = READ_IMAGE_2D(src, sampler, pos).x;
  if (value != 0) {
    value = 0;
  } else {
    value = 1;
  }

  WRITE_IMAGE_2D (dst, pos, value);
}
