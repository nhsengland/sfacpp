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
 * @file ESASingletonStatuses.cpp
 * @brief Singleton class to hold various statuses/messages
 * @date 2025-02-22
 * @author Edmund Haacke
 */

#include "utils/ESASingletonStatuses.hpp"

ESASingletonStatuses* ESASingletonStatuses::pinstance_{nullptr};
std::mutex ESASingletonStatuses::mutex_;

ESASingletonStatuses* ESASingletonStatuses::GetInstance()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pinstance_ == nullptr){
        pinstance_ = new ESASingletonStatuses();
    }
    return pinstance_;
}

void ESASingletonStatuses::setStatus(const std::string& key, const bool value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    stati_[key] = value;
}

bool ESASingletonStatuses::getStatus(const std::string& key) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stati_.find(key) == stati_.end()){
        return false;
    }
    return stati_.at(key);
}

void ESASingletonStatuses::clearStatuses()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stati_.clear();
    stati_int_.clear();
}

void ESASingletonStatuses::setStatusInt(const std::string& key, const int value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    stati_int_[key] = value;
}

std::optional<int> ESASingletonStatuses::getStatusInt(const std::string& key) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stati_int_.find(key) == stati_int_.end()) {
        return std::nullopt;
    }
    return std::make_optional<int>(stati_int_.at(key));
}