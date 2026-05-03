def filtrar_detecciones(detecciones):
    filtradas = []

    for det in detecciones:
        w = det.bbox.maxx - det.bbox.minx
        h = det.bbox.maxy - det.bbox.miny

        if w > 50 or h > 50:
            continue

        filtradas.append(det)

    return filtradas