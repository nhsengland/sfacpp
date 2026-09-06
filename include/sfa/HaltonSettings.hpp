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


#ifndef HALTON_SETTINGS_HPP
#define HALTON_SETTINGS_HPP

struct HaltonSettings {
    // general halton
    int base = 2;
    int start = 7;
    bool useBase = true;
    int burnin = 500;
    bool obsUseSameHaltonDraw = false;
    bool scrambled = true;
    bool shuffle = true;
    // specific to ui0
    int ui0Base = 3;
    int ui0Start = 8;
    bool ui0UseBase = true;
};

#endif // HALTON_SETTINGS_HPP