#include <string>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include "GaussianBlur.h"

using namespace std;

#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define MIN(a,b) ((a)<(b) ? (a) : (b))

double threeway_max(double a, double b, double c) {
    return MAX(a, MAX(b, c));
}

double threeway_min(double a, double b, double c) {
    return MIN(a, MIN(b, c));
}

// r,g,b values are from 0 to 1
// h = [0,360], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)

void RGBtoHSV( float r, float g, float b, float *h, float *s, float *v )
{
    float min, max, delta;
    min = MIN( r, MIN( g, b ));
    max = MAX( r, MAX( g, b ));
    *v = max;				// v
    delta = max - min;
    if( max != 0 )
        *s = delta / max;		// s
    else {
        // r = g = b = 0		// s = 0, v is undefined
        *s = 0;
        *h = -1;
        return;
    }
    if( r == max )
        *h = ( g - b ) / delta;		// between yellow & magenta
    else if( g == max )
        *h = 2 + ( b - r ) / delta;	// between cyan & yellow
    else
        *h = 4 + ( r - g ) / delta;	// between magenta & cyan
    *h *= 60;				// degrees
    if( *h < 0 )
        *h += 360;
}


void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v )
{
    int i;
    float f, p, q, t;
    if( s == 0 ) {
        // achromatic (grey)
        *r = *g = *b = v;
        return;
    }
    h /= 60;			// sector 0 to 5
    i = floor( h );
    f = h - i;			// factorial part of h
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );
    switch( i ) {
        case 0:
            *r = v;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = v;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = v;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = v;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = v;
            break;
        default:		// case 5:
            *r = v;
            *g = p;
            *b = q;
            break;
    }
}

/*
void gimp_cmyk_to_rgb (const GimpCMYK *cmyk, GimpRGB *rgb)

{

  gdouble c, m, y, k;

  k = cmyk->k;


  if (k < 1.0)
    {
      c = cmyk->c * (1.0 - k) + k;
      m = cmyk->m * (1.0 - k) + k;
      y = cmyk->y * (1.0 - k) + k;
    }
  else
    {
      c = 1.0;
      m = 1.0;
      y = 1.0;
    }


  rgb->r = 1.0 - c;
  rgb->g = 1.0 - m;
  rgb->b = 1.0 - y;
  rgb->a = cmyk->a;
}

*/

struct GimpRGB{
    double r;
    double g;
    double b;
    double a;
};

struct GimpHSL{
    double h;
    double s;
    double l;
    double a;
};

double gimp_rgb_max (const GimpRGB *rgb)
{
  if (rgb->r > rgb->g)
    return (rgb->r > rgb->b) ? rgb->r : rgb->b;
  else
    return (rgb->g > rgb->b) ? rgb->g : rgb->b;
}

double gimp_rgb_min (const GimpRGB *rgb)
{
  if (rgb->r < rgb->g)
    return (rgb->r < rgb->b) ? rgb->r : rgb->b;
  else
    return (rgb->g < rgb->b) ? rgb->g : rgb->b;
}


#define GIMP_HSL_UNDEFINED -1.0

void gimp_rgb_to_hsl (const GimpRGB *rgb, GimpHSL *hsl)
{
  double max, min, delta;

  max = gimp_rgb_max (rgb);
  min = gimp_rgb_min (rgb);

  hsl->l = (max + min) / 2.0;

  if (max == min)
    {
      hsl->s = 0.0;
      hsl->h = GIMP_HSL_UNDEFINED;
    }
  else
    {
      if (hsl->l <= 0.5)
        hsl->s = (max - min) / (max + min);
      else
        hsl->s = (max - min) / (2.0 - max - min);

      delta = max - min;

      if (delta == 0.0)
        delta = 1.0;

      if (rgb->r == max)
        {
          hsl->h = (rgb->g - rgb->b) / delta;
        }
      else if (rgb->g == max)
        {
          hsl->h = 2.0 + (rgb->b - rgb->r) / delta;
        }
      else
        {
          hsl->h = 4.0 + (rgb->r - rgb->g) / delta;
        }

      hsl->h /= 6.0;

      if (hsl->h < 0.0)
        hsl->h += 1.0;
    }

  hsl->a = rgb->a;
}


double
gimp_hsl_value (double n1,
                double n2,
                double hue)
{
  double val;

  if (hue > 6.0)
    hue -= 6.0;
  else if (hue < 0.0)
    hue += 6.0;

  if (hue < 1.0)
    val = n1 + (n2 - n1) * hue;
  else if (hue < 3.0)
    val = n2;
  else if (hue < 4.0)
    val = n1 + (n2 - n1) * (4.0 - hue);
  else
    val = n1;

  return val;
}

void
gimp_hsl_to_rgb (const GimpHSL *hsl,
                 GimpRGB       *rgb)
{

  if (hsl->s == 0)
    {
      /*  achromatic case  */
      rgb->r = hsl->l;
      rgb->g = hsl->l;
      rgb->b = hsl->l;
    }
  else
    {
      double m1, m2;

      if (hsl->l <= 0.5)
        m2 = hsl->l * (1.0 + hsl->s);
      else
        m2 = hsl->l + hsl->s - hsl->l * hsl->s;

      m1 = 2.0 * hsl->l - m2;

      rgb->r = gimp_hsl_value (m1, m2, hsl->h * 6.0 + 2.0);
      rgb->g = gimp_hsl_value (m1, m2, hsl->h * 6.0);
      rgb->b = gimp_hsl_value (m1, m2, hsl->h * 6.0 - 2.0);
    }

  rgb->a = hsl->a;
}



int main()
{
    //std::string gaussShaderCode = GenerateGaussFunctionCode(11, false);

    //cout<<gaussShaderCode<<endl;


    /*
    float h, s, l;
    float r = 112, g= 210, b = 137;
    RGBtoHSV(r/255.0, g/255.0, b/255.0, &h, &s, &l);
    cout<<h<<" "<<s<<" "<<l<<endl;

    h = 257;
    s = 0.73;
    l = 0.58;
    HSVtoRGB(&r,&g,&b,h,s,l);
    cout<<r*255<<" "<<g*255<<" "<<b*255<<endl;
    */

    GimpRGB rgb;
    rgb.r = 0.0314;
    rgb.g = 0.1059;
    rgb.b = 0.6941;
    rgb.a = 1.0;

    GimpHSL hsl;
    gimp_rgb_to_hsl(&rgb, &hsl);
    cout<<rgb.r<<" "<<rgb.g<<" "<<rgb.b<<endl;
    cout<<hsl.h<<" "<<hsl.s<<" "<<hsl.l<<endl;
    //hsl.l += 0.1;
    gimp_hsl_to_rgb(&hsl, &rgb);
    cout<<rgb.r<<" "<<rgb.g<<" "<<rgb.b<<endl;
    cout<<hsl.h<<" "<<hsl.s<<" "<<hsl.l<<endl;


	return 0;

}
