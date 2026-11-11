[out:json][timeout:120][bbox:52.36906,9.72934,52.37986,9.74700];
(
  nwr["building"]; nwr["building:part"];
  way["highway"]; nwr["area:highway"]; nwr["place"="square"];
  way["railway"~"^(light_rail|tram|subway|rail|platform)$"];
  node["railway"~"^(subway_entrance|station|stop|tram_stop)$"]; nwr["public_transport"];
  node["natural"="tree"]; way["natural"="tree_row"];
  node["amenity"~"^(bench|clock|fountain|waste_basket|bicycle_parking|drinking_water|post_box|telephone)$"]; way["amenity"="fountain"];
  node["highway"~"^(street_lamp|crossing|traffic_signals|elevator)$"];
  node["advertising"]; node["man_made"]; nwr["tourism"="artwork"]; nwr["historic"];
  nwr["leisure"]; nwr["landuse"]; nwr["barrier"];
);
out body; >; out skel qt;
