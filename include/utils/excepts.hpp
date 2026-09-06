/*
    sfacpp - Estimation of TRE/GTRE SFA models
    Copyright (C) 2025 Edmund Haacke
    Copyright (C) 2025 NHS England

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef ESA_EXCEPTS_HPP
#define ESA_EXCEPTS_HPP

#include <exception>

namespace esaexcepts {
    
    class NotFiniteBase : public std::exception {
    protected:
        const char* _msg;
    public:
        NotFiniteBase(const char* msg) : _msg(msg) {}
        virtual ~NotFiniteBase() = default;
        const char* what() { return _msg; }
    };

    class DensityNotFinite : public NotFiniteBase {
    public:
        DensityNotFinite(const char* msg) : NotFiniteBase(msg) {}
    };

    class GradientNotFinite : public NotFiniteBase {
    public:
        GradientNotFinite(const char* msg) : NotFiniteBase(msg) {}
    };

    class HessianNotFinite : public NotFiniteBase {
    public:
        HessianNotFinite(const char* msg) : NotFiniteBase(msg) {}
    };
}

#endif // ESA_EXCEPTS_HPP