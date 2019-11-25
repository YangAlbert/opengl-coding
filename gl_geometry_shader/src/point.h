#ifndef __SRC_POINT_H__
#define __SRC_POINT_H__

#include "common.h"
#include <cmath>
#include <cfloat>

BEGIN_RENDER_NAMESPACE

struct pointf
{
    float x;
    float y;

    pointf()
        : x(0.f)
        , y(0.f)
    {}

    pointf(float _x, float _y)
        : x(_x)
        , y(_y)
    {
    }

    void set(float _x, float _y)
    {
        x = _x;
        y = _y;
    }

    float length() const
    {
        return std::sqrt(x * x + y * y);
    }

    void normalize()
    {
        float len = length();
        if (len > FLT_EPSILON)
        {
            x /= len;
            y /= len;
        }
    }

    pointf& operator/(float d)
    {
        x /= d;
        y /= d;

        return *this;
    }

    pointf& operator/=(float d)
    {
        *this = *this / d;
        return *this;
    }

    pointf& operator*(float v)
    {
        x *= v;
        y *= v;

        return *this;
    }
};

inline pointf operator+(const pointf& lhs, const pointf& rhs)
{
    pointf pt;
    pt.x = lhs.x + rhs.x;
    pt.y = lhs.y + rhs.y;
    return pt;
}

inline pointf operator-(const pointf& lhs, const pointf& rhs)
{
    pointf pt;
    pt.x = lhs.x - rhs.x;
    pt.y = lhs.y - rhs.y;
    return pt;
}

END_RENDER_NAMESPACE

#endif // __SRC_POINT_H__