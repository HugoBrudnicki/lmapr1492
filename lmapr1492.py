import numpy as np
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.symmetry.bandstructure import HighSymmKpath

def latex_fix(label):
    replace = {
        "\Gamma": "Î“",
    }
    if label in replace:
        label = replace[label]
    return label

# Plotting of the Brillouin zone
def go_points(points, size=4, color="black", labels=None):
    mode = "markers" if labels is None else "markers+text"

    if labels is not None:
        for il in range(len(labels)):
            labels[il] = latex_fix(labels[il])

    import plotly.graph_objects as go
    return go.Scatter3d(
        x=[v[0] for v in points],
        y=[v[1] for v in points],
        z=[v[2] for v in points],
        marker=dict(size=size, color=color),
        mode=mode,
        text=labels,
        textfont_color=color,
        showlegend=False
    )


def go_line(v1, v2, color="black", width=2, mode="lines", text=""):
    import plotly.graph_objects as go
    return go.Scatter3d(
        mode=mode,
        x=[v1[0], v2[0]],
        y=[v1[1], v2[1]],
        z=[v1[2], v2[2]],
        line=dict(color=color),
        text=text,
        showlegend=False
    )

def plot_brillouin_zone(struc, fig=None):
    bz_lattice=struc.lattice.reciprocal_lattice
    if fig is None:
        fig = go.Figure()
#   Plot the three lattice vectors        
    vertex1 = bz_lattice.get_cartesian_coords([0.0, 0.0, 0.0])
    vertex2 = bz_lattice.get_cartesian_coords([1.0, 0.0, 0.0])
    fig.add_trace(go_line(vertex1, vertex2, color="green", mode="lines+text", text=["","a"]))
    vertex2 = bz_lattice.get_cartesian_coords([0.0, 1.0, 0.0])
    fig.add_trace(go_line(vertex1, vertex2, color="green", mode="lines+text", text=["","b"]))
    vertex2 = bz_lattice.get_cartesian_coords([0.0, 0.0, 1.0])
    fig.add_trace(go_line(vertex1, vertex2, color="green", mode="lines+text", text=["","c"]))

#   Plot the Wigner-Seitz cell
    bz = bz_lattice.get_wigner_seitz_cell()
    for iface in range(len(bz)):  # pylint: disable=C0200
        for line in itertools.combinations(bz[iface], 2):
            for jface in range(len(bz)):
                if (iface < jface
                    and any(np.all(line[0] == x) for x in bz[jface])
                    and any(np.all(line[1] == x) for x in bz[jface])):
                    fig.add_trace(go_line(line[0], line[1]))

#   Plot the path in the Brillouin zone
    kpath=HighSymmKpath(struc)
    
    for line in [[kpath.kpath["kpoints"][k] for k in p] for p in kpath.kpath["path"]]:
        for k in range(1, len(line)):
            vertex1 = line[k - 1]
            vertex2 = line[k]
            vertex1 = bz_lattice.get_cartesian_coords(vertex1)
            vertex2 = bz_lattice.get_cartesian_coords(vertex2)

            fig.add_trace(go_line(vertex1, vertex2, color="red"))

    labels=kpath.kpath["kpoints"]
    vecs = []
    for point in labels.values():
        vecs.append(bz_lattice.get_cartesian_coords(point))

    fig.add_trace(go_points(vecs, color="red", labels=list(labels.keys())))

    fig.update_layout(
        scene = dict(
            xaxis = dict(visible=False, range=[-1.15,1.15],),
            yaxis = dict(visible=False, range=[-1.15,1.15],),
            zaxis = dict(visible=False, range=[-1.15,1.15],),
        )
    )
    return fig

# Dealing with the bandstructures
def get_n_branch(bs):
    return len(bs.branches)

def get_n_band(bs):
    if not "phonon" in str(type(bs)):
        return bs.bands[list(bs.bands.keys())[0]].shape[0]
    else:
        return bs.bands.shape[0]
    
def get_branch_wavevectors(bs, i_branch):
    branch = bs.branches[i_branch]
    if not "phonon" in str(type(bs)):
        start_wavevector = bs.kpoints[branch['start_index']].frac_coords
        end_wavevector = bs.kpoints[branch['end_index']].frac_coords
    else:
        start_wavevector = bs.qpoints[branch['start_index']].frac_coords
        end_wavevector = bs.qpoints[branch['end_index']].frac_coords
    return start_wavevector, end_wavevector

def get_branch_labels(bs, i_branch):
    branch = bs.branches[i_branch]
    if not "phonon" in str(type(bs)):
        start_label = bs.kpoints[branch['start_index']].label
        end_label = bs.kpoints[branch['end_index']].label
    else:
        start_label = bs.qpoints[branch['start_index']].label
        end_label = bs.qpoints[branch['end_index']].label        
    start_label = latex_fix(start_label)
    end_label = latex_fix(end_label)
    return [start_label, end_label]

def get_branch_energies(bs, i_branch, i_band):
    branch = bs.branches[i_branch]
    i_start = branch['start_index']
    i_end = branch['end_index']
    if not "phonon" in str(type(bs)):
        energies = list(bs.bands.values())[0][i_band, i_start:i_end+1]
    else:
        energies = bs.bands[i_band, i_start:i_end+1]
    return energies

def get_branch_distances(bs, i_branch):
    branch = bs.branches[i_branch]
    i_start = branch['start_index']
    i_end = branch['end_index']
    distances= bs.distance[i_start:i_end+1]-bs.distance[i_start]*np.ones(i_end-i_start+1)    
    return distances
    
def get_plot_bs(bs, branch_list = "all", plot_range = [None,None]):
    if branch_list == "all":
        branch_list = range(get_n_branch(bs))

    if not "phonon" in str(type(bs)):
        yaxis_title = "E - E<sub>f</sub> (eV)"
        yshift = bs.get_vbm()['energy']
    else:
        yaxis_title = "Frequencies (THz)"
        yshift = 0.0

    if plot_range == [None,None]:
        band_list = range(get_n_band(bs))
    else:
        band_list = []
        for i_band in range(get_n_band(bs)):
            yvals = []
            for i_branch in branch_list:
                yvals.extend(get_branch_energies(bs, i_branch, i_band) - yshift)
            if plot_range[0] == None:
                if np.min(np.array(yvals) - plot_range[1])<=0:
                    band_list.append(i_band)
            elif plot_range[1] == None:
                if np.max(np.array(yvals) - plot_range[0])>=0:
                    band_list.append(i_band)
            else:
                if np.max(np.array(yvals) - plot_range[0])>=0 and np.min(np.array(yvals) - plot_range[1])<=0:
                    band_list.append(i_band)
        
    fig = go.Figure()

    labels = []
    for i_branch in branch_list:
        new_labels = get_branch_labels(bs, i_branch)
        new_xvals = get_branch_distances(bs, i_branch)
        if len(labels) == 0:
            labels.append(new_labels[0])
            xvals = new_xvals.tolist()
            tickvals = [new_xvals[0], new_xvals[-1]]
        elif labels[-1] != new_labels[0]:
            labels[-1] += "|" + new_labels[0]
            xvals.extend((new_xvals + tickvals[-1]).tolist())
            tickvals.append(new_xvals[-1] + tickvals[-1])
        else:
            xvals.extend((new_xvals + tickvals[-1]).tolist())
            tickvals.append(new_xvals[-1] + tickvals[-1])
        labels.append(new_labels[1])
    
    for tickval in tickvals[1:-1]:
        fig.add_vline(x=tickval, line_width=1, line_color="black")

    yvals_lowest = []
    for i_branch in branch_list:
        yvals_lowest.extend(get_branch_energies(bs, i_branch, band_list[0]) - yshift)
    yvals_highest = []
    for i_branch in branch_list:
        yvals_highest.extend(get_branch_energies(bs, i_branch, band_list[-1]) - yshift)

    if plot_range == [None,None]:
        yaxis_range = [
            np.min(yvals_lowest)-0.02*abs(np.min(yvals_lowest)),
            np.max(yvals_highest)+0.02*abs(np.max(yvals_highest))]
    elif plot_range[0] == None:
        yaxis_range = [np.min(yvals_lowest)-0.02*abs(np.min(yvals_lowest)), plot_range[1]]
    elif plot_range[1] == None:
        yaxis_range = [plot_range[0], np.max(yvals_highest)+0.02*abs(np.max(yvals_highest))]
    else:
        yaxis_range = [plot_range[0], plot_range[1]]
        
    for i_band in band_list:
        yvals = []
        for i_branch in branch_list:
            yvals.extend(get_branch_energies(bs, i_branch, i_band) - yshift)
            
        scatter = go.Scatter(x=xvals, y=yvals, mode="lines", name="band "+str(i_band+1))
        fig.add_trace(scatter)
    
    fig.update_layout(
        xaxis =  {'mirror': True, 'showgrid': False,
                  'ticks': 'inside',
                  'tickvals': tickvals,
                  'ticktext': labels,
                  'ticklen':0},
        yaxis =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        yaxis_range = yaxis_range,
        xaxis_title = "Wave Vector",
        yaxis_title = yaxis_title
    )
    
    return fig

def get_plot_dos(dos, plot_range = [None,None]):
    fig = go.Figure()
    if not "phonon" in str(type(dos)):
        xvals = dos.energies - dos.efermi
        xaxis_title = "E - E<sub>f</sub> (eV)"
        yvals = list(dos.densities.values())[0]
    else:
        xvals = dos.frequencies
        xaxis_title = "Frequencies (THz)"
        yvals = dos.densities

    scatter = go.Scatter(x=xvals, y=yvals, mode="lines")
    fig.add_trace(scatter)
    
    if plot_range == [None, None]:
        xaxis_range = [np.min(xvals)-0.02*abs(np.min(xvals)), np.max(xvals)+0.02*abs(np.max(xvals))]
        yaxis_range = [0, 1.02*np.max(yvals)]
    elif plot_range[0] == None:
        xaxis_range = [np.min(xvals)-0.02*abs(np.min(xvals)), plot_range[1]]
        i1 = np.argmin(abs(xvals-plot_range[1]))
        yaxis_range = [0, 1.02*np.max(yvals[:i1])]
    elif plot_range[1] == None:
        xaxis_range = [plot_range[0], np.max(xvals)+0.02*abs(np.max(xvals))]
        i0 = np.argmin(abs(xvals-plot_range[0]))
        yaxis_range = [0, 1.02*np.max(yvals[i0:])]
    else: 
        xaxis_range = [plot_range[0], plot_range[1]]
        i0 = np.argmin(abs(xvals-plot_range[0]))
        i1 = np.argmin(abs(xvals-plot_range[1]))
        yaxis_range = [0, 1.02*np.max(yvals[i0:i1])]
    
    fig.update_layout(
        xaxis =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        yaxis =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        xaxis_title = xaxis_title,
        yaxis_title = "DOS",
    )
    return fig


def get_plot_bs_and_dos(bs, dos, branch_list = "all", plot_range = [None,None]):
    # Bandstructure
    if branch_list == "all":
        branch_list = range(get_n_branch(bs))

    if not "phonon" in str(type(bs)):
        yaxis_title = "E - E<sub>f</sub> (eV)"
        yshift = bs.get_vbm()['energy']
    else:
        yaxis_title = "Frequencies (THz)"
        yshift = 0.0

    if plot_range == [None,None]:
        band_list = range(get_n_band(bs))
    else:
        band_list = []
        for i_band in range(get_n_band(bs)):
            yvals = []
            for i_branch in branch_list:
                yvals.extend(get_branch_energies(bs, i_branch, i_band) - yshift)
            if plot_range[0] == None:
                if np.min(np.array(yvals) - plot_range[1])<=0:
                    band_list.append(i_band)
            elif plot_range[1] == None:
                if np.max(np.array(yvals) - plot_range[0])>=0:
                    band_list.append(i_band)
            else:
                if np.max(np.array(yvals) - plot_range[0])>=0 and np.min(np.array(yvals) - plot_range[1])<=0:
                    band_list.append(i_band)
        
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[4,1])

    labels = []
    for i_branch in branch_list:
        new_labels = get_branch_labels(bs, i_branch)
        new_xvals = get_branch_distances(bs, i_branch)
        if len(labels) == 0:
            labels.append(new_labels[0])
            xvals = new_xvals.tolist()
            tickvals = [new_xvals[0], new_xvals[-1]]
        elif labels[-1] != new_labels[0]:
            labels[-1] += "|" + new_labels[0]
            xvals.extend((new_xvals + tickvals[-1]).tolist())
            tickvals.append(new_xvals[-1] + tickvals[-1])
        else:
            xvals.extend((new_xvals + tickvals[-1]).tolist())
            tickvals.append(new_xvals[-1] + tickvals[-1])
        labels.append(new_labels[1])
    
    yvals_lowest = []
    for i_branch in branch_list:
        yvals_lowest.extend(get_branch_energies(bs, i_branch, band_list[0]) - yshift)
    yvals_highest = []
    for i_branch in branch_list:
        yvals_highest.extend(get_branch_energies(bs, i_branch, band_list[-1]) - yshift)

    if plot_range == [None,None]:
        yaxis_range = [
            np.min(yvals_lowest)-0.02*abs(np.min(yvals_lowest)),
            np.max(yvals_highest)+0.02*abs(np.max(yvals_highest))]
    elif plot_range[0] == None:
        yaxis_range = [np.min(yvals_lowest)-0.02*abs(np.min(yvals_lowest)), plot_range[1]]
    elif plot_range[1] == None:
        yaxis_range = [plot_range[0], np.max(yvals_highest)+0.02*abs(np.max(yvals_highest))]
    else:
        yaxis_range = [plot_range[0], plot_range[1]]
        
    for i_band in band_list:
        yvals = []
        for i_branch in branch_list:
            yvals.extend(get_branch_energies(bs, i_branch, i_band) - yshift)
        scatter = go.Scatter(x=xvals, y=yvals, mode="lines", name="band "+str(i_band+1))
        fig.add_trace(scatter, row=1, col=1)

    for tickval in tickvals[1:-1]:
        fig.add_vline(x=tickval, line_width=1, line_color="black", row=1, col=1)
    
    # DOS
    if not "phonon" in str(type(dos)):
        xvals2 = list(dos.densities.values())[0]
        yvals2 = dos.energies - dos.efermi
    else:
        xvals2 = dos.densities
        yvals2 = dos.frequencies

    scatter = go.Scatter(x=xvals2, y=yvals2, mode="lines", showlegend=False)
    fig.add_trace(scatter, row=1, col=2)

    i0 = np.argmin(abs(yvals2-yaxis_range[0]))
    i1 = np.argmin(abs(yvals2-yaxis_range[1]))
    xaxis2_range = [0, 1.02*np.max(xvals2[i0:i1])]

    fig.update_layout(
        xaxis =  {'mirror': True, 'showgrid': False,
                  'ticks': 'inside',
                  'tickvals': tickvals,
                  'ticktext': labels,
                  'ticklen':0},
        yaxis =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        yaxis_range = yaxis_range,
        xaxis_title = "Wave Vector",
        yaxis_title = yaxis_title,        
        xaxis2 =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        yaxis2 =  {'mirror': True, 'showgrid': False, 'ticks': 'inside', 'ticklen':10},
        xaxis2_range = xaxis2_range,
        xaxis2_title = "DOS",
    )
    
    return fig