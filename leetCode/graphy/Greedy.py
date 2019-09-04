

def cover_states(states_needed, stations):
    stations_selected = set()
    while states_needed:
        best_station = None
        states_covered = set()
        for station, station_cover_states in stations.items():
            covered = states_needed & station_cover_states
            # 保证有每次新增的states有多覆盖就ok
            if len(covered) > len(states_covered):
                best_station = station
                states_covered = covered
        states_needed = states_needed - states_covered
        stations_selected.add(best_station)
    return stations_selected


if __name__ == "__main__":
    states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])
    stations = {}
    stations["Kone"] = set(["id", "nv", "ut"])
    stations["ktwo"] = set(["wa", "id", "mt"])
    stations["kthree"] = set(["or", "nv", "ca"])
    stations["kfour"] = set(["nv", "ut"])
    stations["kfive"] = set(["ca", "az"])
    print(cover_states(states_needed, stations))