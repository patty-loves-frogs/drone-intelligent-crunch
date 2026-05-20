def reporter_node(state):
    """Generate a comprehensive report from YOLO and VLM analysis results."""
    detections = state.get("detections", [])
    vlm_observations = state.get("vlm_observations", [])
    
    yolo_json = state.get("yolo_json", {})
    summary = yolo_json.get("summary", {})

    # Résumé YOLO enrichi
    yolo_lines = []
    for d in detections:
        if d.get("label") == "error":
            continue

        yolo_lines.append(
            f"- frame={d.get('frame_id')}, image={d.get('image_path')}: "
            f"{d.get('label')}, posture={d.get('posture', 'non précisée')}, "
            f"conf={d.get('confidence')}, bbox={d.get('bounding_box')}"
        )

    yolo_summary = "\n".join(yolo_lines) if yolo_lines else "Aucune détection YOLO."

    video_summary = (
        f"Vidéo : {yolo_json.get('video_path', 'non précisée')}\n"
        f"Frames analysées : {yolo_json.get('frames_analyzed', 0)}\n"
        f"Personnes détectées : {summary.get('persons_detected', 0)}\n"
        f"Debout : {summary.get('standing_count', 0)}\n"
        f"Allongées : {summary.get('lying_count', 0)}\n"
        f"Incertaines : {summary.get('uncertain_count', 0)}\n"
        f"Faible confiance : {summary.get('low_confidence_count', 0)}"
    )

    # Résumé VLM enrichi
    vlm_lines = []
    for obs in vlm_observations:
        vlm_lines.append(
            f"- {obs.get('image_path')}: "
            f"person_visible={obs.get('person_visible')}, "
            f"confidence={obs.get('confidence')}, "
            f"text={obs.get('text')}, "
            f"error={obs.get('error')}"
        )

    vlm_summary = "\n".join(vlm_lines) if vlm_lines else "Aucune analyse VLM effectuée."
    
    # Combine summaries into final report
    report = f"{video_summary}\n\n{yolo_summary}\n\n{vlm_summary}"
    
    return {"report": report}