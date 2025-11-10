#!/bin/bash

echo " Setting up highD SUMO simulation..."

# Create necessary directories
echo " Creating directories..."
mkdir -p sumo_cfg
mkdir -p sumo_cfg/data
mkdir -p data/highd/dataset
mkdir -p veh_config

# Copy or update the network file
echo "  Setting up network file..."
cat > sumo_cfg/net.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="-1000.00,0.00,5000.00,100.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="E0" from="J0" to="J1" priority="-1" length="6000.00">
        <lane id="E0_0" index="0" speed="44.44" length="6000.00" shape="-1000.00,7.00 5000.00,7.00"/>
        <lane id="E0_1" index="1" speed="44.44" length="6000.00" shape="-1000.00,10.20 5000.00,10.20"/>
        <lane id="E0_2" index="2" speed="44.44" length="6000.00" shape="-1000.00,13.40 5000.00,13.40"/>
    </edge>
    <junction id="J0" type="dead_end" x="-1000.00" y="10.00" incLanes="" intLanes="" shape="-1000.00,14.80 -1000.00,5.60"/>
    <junction id="J1" type="dead_end" x="5000.00" y="10.00" incLanes="E0_0 E0_1 E0_2" intLanes="" shape="5000.00,5.60 5000.00,14.80"/>
</net>
EOF

# Setup route file
echo "  Setting up routes..."
cat > sumo_cfg/route.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <route id="platoon_route" edges="E0"/>
    <route id="merge_route" edges="E0"/>
</routes>
EOF

# Setup configuration file
echo "  Setting up configuration..."
cat > sumo_cfg/freeway.sumo.cfg << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
  <input>
    <net-file value="net.xml"/>
    <route-files value="route.xml"/>
    <additional-files value="vTypeDistributions.add.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="360000"/>
    <step-length value="0.25"/>
  </time>
  <processing>
    <collision.action value="warn"/>
    <collision.stoptime value="10"/>
    <time-to-teleport value="-1"/>
    <max-depart-delay value="600"/>
  </processing>
  <report>
    <verbose value="true"/>
    <no-step-log value="false"/>
  </report>
  <gui_only>
    <start value="true"/>
    <quit-on-end value="false"/>
    <gui-settings-file value="gui-settings.xml"/>
  </gui_only>
</configuration>
EOF

# Setup GUI settings
echo "Setting up GUI settings..."
cat > sumo_cfg/gui-settings.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world">
        <opengl antialiase="1" dither="1"/>
    </scheme>
    <viewport zoom="500" x="2000" y="10" angle="0"/>
    <delay value="20"/>
</viewsettings>
EOF

# Generate vehicle type distributions if script exists
if [ -f "geneType.sh" ]; then
    echo " Generating vehicle types..."
    bash geneType.sh
else
    echo "  Warning: geneType.sh not found. You'll need to generate vehicle types manually."
    echo "   Run: bash geneType.sh"
fi

# Check for highD dataset
if [ ! -f "data/highd/dataset/02_tracksMeta.csv" ] || [ ! -f "data/highd/dataset/02_tracks.csv" ]; then
    echo ""
    echo " WARNING: highD dataset files not found!"
    echo "   Please download the highD dataset and place the following files:"
    echo "   - data/highd/dataset/02_tracksMeta.csv"
    echo "   - data/highd/dataset/02_tracks.csv"
    echo ""
    echo "   Download from: https://www.highd-dataset.com/"
    echo ""
fi

echo ""
echo " Setup complete!"
echo ""
echo "To run the simulation:"
echo "  1. Ensure highD dataset files are in data/highd/dataset/"
echo "  2. Generate vehicle types: bash geneType.sh"
echo "  3. Run simulation: python3 main.py"
echo ""
echo "To view the network in SUMO-GUI:"
echo "  sumo-gui -c sumo_cfg/freeway.sumo.cfg"
echo ""