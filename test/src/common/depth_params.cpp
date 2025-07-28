#include <cstdio>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include "common/depth_params.h"

namespace common
{
    DepthParams DepthParams::from_json(const std::string &json_path)
    {
        DepthParams params;
        FILE *fp = fopen(json_path.c_str(), "r");
        if (!fp)
        {
            std::cerr << "Cannot open file: " << json_path << std::endl;
            throw std::runtime_error("Cannot open json file");
        }
        char readBuffer[65536];
        rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
        rapidjson::Document doc;
        doc.ParseStream(is);
        fclose(fp);
        if (!doc.IsObject())
        {
            throw std::runtime_error("JSON root is not an object");
        }
        if (doc.HasMember("name"))
            params.name = doc["name"].GetString();
        if (doc.HasMember("clip_start"))
            params.clip_start = doc["clip_start"].GetDouble();
        if (doc.HasMember("clip_end"))
            params.clip_end = doc["clip_end"].GetDouble();
        if (doc.HasMember("max_val_in_depth"))
            params.max_val_in_depth = doc["max_val_in_depth"].GetDouble();
        if (doc.HasMember("num_input_points"))
            params.num_input_points = doc["num_input_points"].GetInt();
        return params;
    }
}
