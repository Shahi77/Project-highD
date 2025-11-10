#!/bin/bash

set -e

echo " Generating vehicle type distributions..."

if [ -z "$SUMO_HOME" ]; then
    echo " Error: SUMO_HOME is not set"
    exit 1
fi

if [ ! -f "$SUMO_HOME/tools/createVehTypeDistribution.py" ]; then
    echo " Error: createVehTypeDistribution.py not found"
    exit 1
fi

if [ ! -f "sumo_cfg/veh_config/car.config.txt" ]; then
    echo " Error: sumo_cfg/veh_config/car.config.txt missing"
    exit 1
fi

if [ ! -f "sumo_cfg/veh_config/truck.config.txt" ]; then
    echo " Error: sumo_cfg/veh_config/truck.config.txt missing"
    exit 1
fi

rm -f vTypeDistributions.add.xml

echo " Generating 1000 car types..."
python3 "$SUMO_HOME/tools/createVehTypeDistribution.py" \
    sumo_cfg/veh_config/car.config.txt \
    --size 1000 \
    --name car

echo " Generating 1000 truck types..."
python3 "$SUMO_HOME/tools/createVehTypeDistribution.py" \
    sumo_cfg/veh_config/truck.config.txt \
    --size 1000 \
    --name truck

if command -v sed &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '.bak' 's/\.000//g' vTypeDistributions.add.xml
        rm -f vTypeDistributions.add.xml.bak
    else
        sed -i 's/\.000//g' vTypeDistributions.add.xml
    fi
fi

cp vTypeDistributions.add.xml sumo_cfg/

echo " Done"
