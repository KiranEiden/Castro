def _soundspeed(field, data):
    return (4./3. * data['pressure'] / data['density'])**0.5

def _magvel(field, data):
    return (data['x_velocity']**2 + data['y_velocity']**2)**0.5

add_field("soundspeed", function=_soundspeed, sampling_type="cell",
          dimensions="velocity", units="auto")
add_field("magvel", function=_magvel, sampling_type="cell",
          dimensions="velocity", units="auto")
