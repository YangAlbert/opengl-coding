#ifndef __SRC_CONTEXT_WRAPPER_H__
#define __SRC_CONTEXT_WRAPPER_H__

#include "common.h"
#include <string>

BEGIN_RENDER_NAMESPACE

class RenderWindow
{
    RenderWindow();
    virtual ~RenderWindow();

    virtual bool CreateWindow(const std::string& wind_name, int width, int height);
}

END_RENDER_NAMESPACE

#endif // __SRC_CONTEXT_WRAPPER_H__