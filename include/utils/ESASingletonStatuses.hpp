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

/**
 * @file ESASingletonStatuses.hpp
 * @brief Singleton class to hold various statuses/messages
 * @date 2025-02-22
 * @author Edmund Haacke
 */

#ifndef ESA_SINGLETON_STATUSES_HPP
#define ESA_SINGLETON_STATUSES_HPP

#include <map>
#include <mutex>
#include <string>
#include <optional>

class ESASingletonStatuses {

private:
    static ESASingletonStatuses* pinstance_;
    static std::mutex mutex_;

    std::map<std::string, bool> stati_;
    std::map<std::string, int> stati_int_;

protected:
    ESASingletonStatuses(){};
    ~ESASingletonStatuses(){};

public:

    // remove ability to clone
    ESASingletonStatuses(ESASingletonStatuses& other) = delete;
    // no ability to assign to signleton
    void operator=(const ESASingletonStatuses&) = delete;
    // static method to control access to singleton
    static ESASingletonStatuses* GetInstance();

    void setStatus(const std::string& key, const bool value);
    bool getStatus(const std::string& key) const;
    void setStatusInt(const std::string& key, const int value);
    std::optional<int> getStatusInt(const std::string& key) const;
    void clearStatuses();

};

#endif // ESA_SINGLETON_STATUSES_HPP