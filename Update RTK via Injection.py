# What
What = 0
import deap


def oldupdate_rtk_parameters(
    inp_file_path="C:/Users/chase/Downloads/TEST.inp",
    hydrograph_name="Analysis_2008-03-03",
    new_rtk_list=[(1, 2, 3), (4, 5, 6), (7, 7, 9)],
):
    """
    Replaces RTK triplets for a given hydrograph name.
    - inp_file_path: Path to the .inp file
    - hydrograph_name: Name of the hydrograph to update
    - new_rtk_list: List of (R, T, K) tuples, max 3
    """
    updated_lines = []
    in_hydrograph_section = False
    skip_next = 0

    with open(inp_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if stripped.upper() == "[HYDROGRAPHS]":
            in_hydrograph_section = True
            updated_lines.append(line)
            continue

        if in_hydrograph_section:
            if stripped == "" or stripped.startswith("["):
                # end of section or start of next
                in_hydrograph_section = False
                # Check if the line has
                # insert updated RTK values
                for rtk in new_rtk_list:
                    r, t, k = rtk
                    updated_lines.append(f"{hydrograph_name}   {r}   {t}   {k}\n")
                updated_lines.append(line)
                continue

            # Skip lines with the matching hydrograph
            if stripped.startswith(hydrograph_name):
                continue  # Don't add existing lines
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    with open("C:/Users/chase/Downloads/TESTMODIFIED.inp", "w") as f:
        f.writelines(updated_lines)


update_rtk_columns()
