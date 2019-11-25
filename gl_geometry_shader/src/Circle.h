#ifndef __SRC_CIRCLE_H__
#define __SRC_CIRCLE_H__

#include "common.h"
#include "point.h"

BEGIN_RENDER_NAMESPACE

struct Circle
{
    // for rendering.
    pointf position;

    // current life.
    float life;

    static const float kStartLife;
    static const float kStartRad;
    static const float kEndRad;
    static const float kStartColor[4];
    static const float kEndColor[4];

    static const int kPositionOffset = 0;
    static const int kLifeOffset = sizeof(pointf);
    //static const int kRadiusOffset = sizeof(pointf) + sizeof(color);

    Circle()
        : Circle({0.0f, 0.0f})
    {}

    Circle(const pointf& pos)
        : position(pos)
        //, color{0.8f, 0.8f, 0.8f, 1.0f}
        , life(kStartLife)
    {
    }

    float normalizedLife() const
    {
        return life / kStartLife;
    }
};

END_RENDER_NAMESPACE

#endif // __SRC_CIRCLE_H__