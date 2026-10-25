%% Shuttle-Glider tracking - flight regimes and the mission with the PX4 estimator
% Data: data/*.csv written by export_results.py.
% All text uses the LaTeX interpreter, i.e. Computer Modern.
% Variants: Base, convex filtered (wls_ema) and convex filtered + lateral moments (wls_ema_m02, retained), seed 3.
%
%   plot_results                 % all figures, saved to figures/ (PDF + PNG)
%   plot_results(false)          % draw only
%
% Figures, one file per panel (15 x 10 cm, as the episode panels of the Shuttle chapters), placed side by side in
% LaTeX (subfigures, 0.49\linewidth) with the common legend file above:
%   fig01a..g_decel_v9_*    deceleration 9 -> 0 m/s, one episode per variant from the trim without perturbation, one
%                           quantity per axes and the legend of the variants on top of every file: a flight state
%                           (speed and reference, pitch, angle of attack), b position / velocity error, c lift total /
%                           puller thrust, d moment x / y / z of the lift rotors, e aerodynamic force x / y /
%                           z and f moment x / y / z in body axes, g elevator / aileron / rudder
%   fig02a..g_helix_v12_*   helical turn at 12 m/s, R = 30 m, 1 m/s climb, same panels (a: bank, alpha, sideslip)
%   fig01_legend            variants and reference, placed above the subfigures (the panels have no legend)
%   fig01h, fig02h          beside the surfaces: shaft power and wing lift (deceleration), 3D path (helix)
%   fig03_mission_px4       one figure at the text width (16 x 10.5 cm): mission at 12 m/s of the retained policy (M) with
%                           the true state, with the PX4 EKF2 estimate, and with the estimate and a GPS without random
%                           walk: speed (true-state flight) and estimated speed (PX4 flights), phases shaded, error of
%                           the true position (log), error of the estimated
%                           position (the error the policy observes); legend on top
%   figA3_mission_px4_base  appendix: as fig03 for the base policy
%   figA4_mission_path_base appendix: as figA2 for the base policy
%   fig04_steady_alpha_power  text width (16 x 7 cm): angle of attack and power relative to the minimum-power trim
%                           against speed in level cruise at 1 to 14 m/s, the three variants (trim dashed)
%   figA1_helix_path        appendix: 3D path of the helix, the three variants and the reference
%   figA2_mission_path      path of the mission in the horizontal plane (reference coloured by its speed) and altitude
%                           against time, with the true position of the two PX4 flights
% Requires MATLAB R2020b or later.

function plot_results(saveFigures)
if nargin < 1, saveFigures = true; end
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
figDir  = fullfile(here, 'figures');
if saveFigures && ~exist(figDir, 'dir'), mkdir(figDir); end

% Computer Modern: every text object through the LaTeX interpreter
names = {'defaultTextInterpreter', 'defaultAxesTickLabelInterpreter', 'defaultLegendInterpreter', ...
         'defaultColorbarTickLabelInterpreter', 'defaultAxesFontName', 'defaultTextFontName', ...
         'defaultLegendFontName', 'defaultAxesFontSize', 'defaultTextFontSize', 'defaultLegendFontSize'};
previous = cellfun(@(n) get(groot, n), names, 'UniformOutput', false);
for k = 1:4, set(groot, names{k}, 'latex'); end      % the first four are interpreters
set(groot, 'defaultAxesFontName', 'Computer Modern', 'defaultTextFontName', 'Computer Modern', ...
           'defaultLegendFontName', 'Computer Modern', 'defaultAxesFontSize', 9, 'defaultTextFontSize', 9, ...
           'defaultLegendFontSize', 9);
restore = onCleanup(@() restoreDefaults(names, previous));

S = style();

%% fig01, fig02 - one episode per variant, one file per panel (placed as subfigures in LaTeX)
regimes = ["decel_v9_0", "helix_v12_r30_z1"];
prefix  = ["fig01", "fig02"];
shorts  = ["decel_v9", "helix_v12"];
for r = 1:numel(regimes)
    EP = readtable(fullfile(dataDir, "episode_" + regimes(r) + ".csv"), 'TextType', 'string');
    E = cell(1, numel(S.variants));
    for i = 1:numel(S.variants), E{i} = EP(EP.variant == S.variants(i), :); end
    base = prefix(r) + "%s_" + shorts(r) + "_%s";

    one = {'-'};
    % a: flight state - deceleration: speed and reference, pitch (to brake), alpha; helix: bank, alpha and sideslip
    % (the speed is constant within 2 cm/s there, the sideslip shows the crossed controls)
    if startsWith(regimes(r), "decel")
        panel(S, E, {{'v_along'}, {'pitch'}, {'alpha'}}, {one, one, one}, [1 1 1], ...
              {'$V$ [m/s]', '$\theta$ [deg]', '$\alpha$ [deg]'}, figDir, sprintf(base, "a", "flight"), saveFigures, ...
              {'v_along_ref', '', ''});
    else
        panel(S, E, {{'roll'}, {'alpha'}, {'beta'}}, {one, one, one}, [1 1 1], ...
              {'$\phi$ [deg]', '$\alpha$ [deg]', '$\beta$ [deg]'}, figDir, sprintf(base, "a", "flight"), saveFigures);
    end
    panel(S, E, {{'e_p'}, {'e_v'}}, {one, one}, [100 1], {'$\|\mathbf{e}_p\|$ [cm]', '$\|\mathbf{e}_v\|$ [m/s]'}, ...
          figDir, sprintf(base, "b", "errors"), saveFigures);
    panel(S, E, {{'T_lift'}, {'Tp'}}, {one, one}, [1 1], {'$\sum T_i$ [N]', '$T_p$ [N]'}, ...
          figDir, sprintf(base, "c", "thrust"), saveFigures);
    panel(S, E, {{'Mx_rot'}, {'My_rot'}, {'Mz_rot'}}, {one, one, one}, [1 1 1], ...
          {'$M_{x,r}$ [N\,m]', '$M_{y,r}$ [N\,m]', '$M_{z,r}$ [N\,m]'}, figDir, sprintf(base, "d", "rotor_moment"), saveFigures);
    panel(S, E, {{'Fx'}, {'Fy'}, {'Fz'}}, {one, one, one}, [1 1 1], {'$F_x$ [N]', '$F_y$ [N]', '$F_z$ [N]'}, ...
          figDir, sprintf(base, "e", "aero_force"), saveFigures);
    panel(S, E, {{'Mx'}, {'My'}, {'Mz'}}, {one, one, one}, [1 1 1], ...
          {'$M_x$ [N\,m]', '$M_y$ [N\,m]', '$M_z$ [N\,m]'}, figDir, sprintf(base, "f", "aero_moment"), saveFigures);
    panel(S, E, {{'de'}, {'da'}, {'dr'}}, {one, one, one}, [1 1 1], ...
          {'$\delta_e$ [deg]', '$\delta_a$ [deg]', '$\delta_r$ [deg]'}, figDir, sprintf(base, "g", "surfaces"), saveFigures);
    % h: beside the surfaces - shaft power and wing lift in the deceleration (the transition from wing to rotors),
    % 3D path in the helix
    if startsWith(regimes(r), "decel")
        panel(S, E, {{'power'}, {'lift'}}, {one, one}, [1 1], {'$P$ [W]', '$L$ [N]'}, ...
              figDir, sprintf(base, "h", "power_lift"), saveFigures);
    else
        pathPanel(S, E, figDir, sprintf(base, "h", "path"), saveFigures);
    end
end
legendFile(S, [S.labels, {'Reference'}], [num2cell(S.color, 2); {[0 0 0]}], [S.lineStyle, {':'}], figDir, ...
           'fig01_legend', saveFigures);

%% fig04 - steady flight: angle of attack and power against speed, the three variants and the minimum-power trim
D = readtable(fullfile(dataDir, 'steady_alpha_power.csv'), 'TextType', 'string');
T = readtable(fullfile(dataDir, 'trim_curve.csv'));
f = newFigure(S, S.textW, 7);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ylabs = {'$\alpha$ [deg]', '$(P - P_{\mathrm{trim}}) / P_{\mathrm{trim}}$ [\%]'};
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    if k == 1
        ht = plot(ax, T.speed, T.alpha_deg, '--', 'Color', 'k', 'LineWidth', 1.0);
    else
        plot(ax, [0 15], [0 0], '--', 'Color', 'k', 'LineWidth', 1.0);      % the trim itself
    end
    hv = gobjects(1, numel(S.variants));
    for i = 1:numel(S.variants)
        Di = sortrows(D(D.variant == S.variants(i), :), 'speed');
        if k == 1
            y = Di.alpha_deg;
        else
            y = 100 * (Di.power_W ./ interp1(T.speed, T.power_W, Di.speed) - 1);
        end
        hv(i) = plot(ax, Di.speed, y, [S.lineStyle{i} S.marker{i}], 'Color', S.color(i, :), ...
                     'MarkerFaceColor', S.color(i, :), 'MarkerSize', 4, 'LineWidth', 1.0);
    end
    xlim(ax, [0 15]); xlabel(ax, '$V$ [m/s]'); ylabel(ax, ylabs{k}); formatAxis(ax, S);
end
lgd = legend([hv, ht], [S.labels, {'Trim (minimum power)'}], 'Orientation', 'horizontal', 'Box', 'off', ...
             'Interpreter', 'latex', 'FontSize', 8);
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, 'fig04_steady_alpha_power', saveFigures);

%% fig03 (filtered + lateral moments, retained) and figA3 (base, appendix) - mission with the PX4 estimator
missionFigure(S, dataDir, 'mission_v12', figDir, 'fig03_mission_px4', saveFigures);
missionFigure(S, dataDir, 'mission_v12_base', figDir, 'figA3_mission_px4_base', saveFigures);

%% figA1 - 3D path of the helix (appendix): reference (black dotted) and the three variants
EP = readtable(fullfile(dataDir, 'episode_helix_v12_r30_z1.csv'), 'TextType', 'string');
E1 = EP(EP.variant == S.variants(1), :);
f = newFigure(S, S.textW, 9);
ax = axes(f); hold(ax, 'on');
% Reference drawn last and in child order: with depth sorting the -painters PDF hides it under the variants
set(ax, 'SortMethod', 'childorder');
hv = gobjects(1, numel(S.variants));
for i = 1:numel(S.variants)
    Ei = EP(EP.variant == S.variants(i), :);
    hv(i) = plot3(ax, Ei.x, Ei.y, Ei.z, S.lineStyle{i}, 'Color', S.color(i, :), 'LineWidth', 1.0);
end
hr = plot3(ax, E1.x_ref, E1.y_ref, E1.z_ref, ':', 'Color', 'k', 'LineWidth', 1.1);
view(ax, -54, 12.6); daspect(ax, [1 1 0.25]);
xlabel(ax, '$x$ [m]'); ylabel(ax, '$y$ [m]'); zlabel(ax, '$z$ [m]');
formatAxis(ax, S);
lgd = legend(ax, [hr, hv], [{'Reference'}, S.labels], 'Location', 'northoutside', 'Orientation', 'horizontal', ...
       'Box', 'off', 'Interpreter', 'latex');
legendOnDrawing(ax, lgd, []);
saveFigure(f, figDir, 'figA1_helix_path', saveFigures);

%% figA2 (filtered + lateral moments, retained) and figA4 (base) - 3D path of the mission (appendix)
missionPath(S, dataDir, 'mission_v12', figDir, 'figA2_mission_path', saveFigures);
missionPath(S, dataDir, 'mission_v12_base', figDir, 'figA4_mission_path_base', saveFigures);
end

%% =============================================================== helpers
function restoreDefaults(names, values)
for k = 1:numel(names), set(groot, names{k}, values{k}); end
end

function S = style()
S.font = 9;
S.textW = 16;                                         % [cm] text width (full-width figures)
S.panelW = 15; S.panelH = 8.5;                        % [cm] one panel; 4 rows of 0.49\linewidth fit on a page
S.panelLegend = false;                                % legend of the variants in a separate file (fig01_legend)
S.variants = ["base", "wls_ema", "wls_ema_m02"];
S.labels = {'Base', 'Convex, filtered', 'Convex, filtered + moments'};
S.color = [0.165 0.471 0.839;    % base          (#2a78d6, categorical slot 1)
           0.922 0.408 0.204;    % wls_ema       (#eb6834, slot 2)
           0.106 0.686 0.478];   % wls_ema_m02   (#1baf7a, slot 3)
S.lineStyle = {'-', '--', '-.'};                      % per variant, in addition to the colour
S.marker = {'o', 's', '^'};                           % per variant (steady-flight figure)
% Multi-hue sequential colour map for the reference speed (dark blue at rest -> yellow at cruise speed)
S.speedMap = parula(256);
S.speedRange = [0 12];                                % [m/s] colour scale of the reference speed (mission)
S.cases = ["true", "px4", "px4_nodrift"];
S.caseLabels = {'True state', 'PX4 estimate', 'PX4 estimate, GPS without drift'};
S.caseColor = [0.25 0.25 0.25;   % true state (neutral)
               0.922 0.408 0.204;
               0.165 0.471 0.839];
end

function missionPath(S, dataDir, name, figDir, file, doSave)
% Path of a mission in two views at the text width: (a) horizontal plane and (b) altitude against time, the
% reference coloured by its speed in both. The true position of the two PX4 flights in dark grey (solid: PX4, dashed: GPS without
% drift); the flight with the true state and the estimated positions lie on the reference within centimetres.
% Two 2D views instead of one 3D view: the path spans about 520 x 100 m but only 10 m in height.
M = readtable(fullfile(dataDir, [name '.csv']), 'TextType', 'string');
P = readtable(fullfile(dataDir, [name '_phases.csv']), 'TextType', 'string');
ink = [0.25 0.25 0.25];
f = newFigure(S, S.textW, 9);
tl = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl); hold(ax, 'on');
h2 = plot(ax, M.x_px4, M.y_px4, '-', 'Color', ink, 'LineWidth', 0.9);
h3 = plot(ax, M.x_px4_nodrift, M.y_px4_nodrift, '--', 'Color', ink, 'LineWidth', 0.9);
hr = surface(ax, [M.x_ref M.x_ref], [M.y_ref M.y_ref], zeros(height(M), 2), [M.V_ref M.V_ref], ...
             'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 3.0);
uistack(hr, 'bottom');                                % the flown paths stay visible on top of the thick reference
colormap(ax, S.speedMap); clim(ax, S.speedRange);
cb = colorbar(ax);
cb.Layout.Tile = 'east';                              % beside both views, so that their widths match
cb.Label.String = '$V_d$ [m/s]'; cb.Label.Interpreter = 'latex'; cb.TickLabelInterpreter = 'latex';
axis(ax, 'equal'); view(ax, 2);
xlabel(ax, '$x$ [m]'); ylabel(ax, '$y$ [m]');
formatAxis(ax, S);

ax = nexttile(tl); hold(ax, 'on');
% reference drawn as in the horizontal view (thick, coloured by its speed), so that one legend entry covers both
hz = surface(ax, [M.t M.t], [M.z_ref M.z_ref], zeros(height(M), 2), [M.V_ref M.V_ref], ...
             'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 3.0);
colormap(ax, S.speedMap); clim(ax, S.speedRange);
plot(ax, M.t, M.z_px4, '-', 'Color', ink, 'LineWidth', 0.9);
plot(ax, M.t, M.z_px4_nodrift, '--', 'Color', ink, 'LineWidth', 0.9);
yl = [floor(min([M.z_ref; M.z_px4; M.z_px4_nodrift])) - 1, ceil(max([M.z_ref; M.z_px4; M.z_px4_nodrift])) + 1];
ylim(ax, yl);
phaseBands(ax, P, yl, false);
timeAxis(ax, S, '$z$ [m]', true, [0 M.t(end)]);
uistack(hz, 'bottom'); uistack(findobj(ax, 'Type', 'patch'), 'bottom');   % bands, reference, flights

lgd = legend([hr, h2, h3], {'Reference (colour: speed)', 'PX4: true position', ...
       'PX4, GPS without drift: true position'}, 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex');
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, file, doSave);
end

function missionFigure(S, dataDir, name, figDir, file, doSave)
% Mission with the true state, the PX4 estimate and the estimate with a GPS without random walk: speed (phases
% shaded), error of the true position (log), error of the estimated position; legend on top.
M = readtable(fullfile(dataDir, [name '.csv']), 'TextType', 'string');
P = readtable(fullfile(dataDir, [name '_phases.csv']), 'TextType', 'string');
tEnd = M.t(end);
f = newFigure(S, S.textW, 10.5);                    % wide and short: three stacked axes at the text width
tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl); hold(ax, 'on');
% Speed of the true-state flight and speed estimated by the EKF in the two PX4 flights
plot(ax, M.t, M.V_ref, ':', 'Color', 'k', 'LineWidth', 1.1);
plot(ax, M.t, M.V, '-', 'Color', S.caseColor(1, :), 'LineWidth', 1.2);
for c = 2:numel(S.cases)
    plot(ax, M.t, M.(char("Ve_" + S.cases(c))), '-', 'Color', S.caseColor(c, :), 'LineWidth', 1.2);
end
ylim(ax, [0 14]);
phaseBands(ax, P, [0 14], true);
timeAxis(ax, S, '$V$, $\hat V$ [m/s]', false, [0 tEnd]);

ax = nexttile(tl); hold(ax, 'on');
h = gobjects(1, numel(S.cases));
for c = 1:numel(S.cases)
    h(c) = plot(ax, M.t, 100 * M.(char("e_true_" + S.cases(c))), '-', 'Color', S.caseColor(c, :), 'LineWidth', 1.2);
end
set(ax, 'YScale', 'log'); ylim(ax, [0.1 1000]); yticks(ax, [0.1 1 10 100 1000]);
phaseBands(ax, P, [0.1 1000], false);
timeAxis(ax, S, '$\|\mathbf{p} - \mathbf{p}_d\|$ [cm]', false, [0 tEnd]);

ax = nexttile(tl); hold(ax, 'on');
for c = 2:numel(S.cases)
    plot(ax, M.t, 100 * M.(char("e_est_" + S.cases(c))), '-', 'Color', S.caseColor(c, :), 'LineWidth', 1.2);
end
ylim(ax, [0 30]);
phaseBands(ax, P, [0 30], false);
timeAxis(ax, S, '$\|\hat{\mathbf{p}} - \mathbf{p}_d\|$ [cm]', true, [0 tEnd]);

href = plot(ax, NaN, NaN, ':', 'Color', 'k', 'LineWidth', 1.1);
lgd = legend([h, href], [S.caseLabels, {'Reference'}], 'Orientation', 'horizontal', 'Box', 'off', ...
             'Interpreter', 'latex');
lgd.Layout.Tile = 'north';
saveFigure(f, figDir, file, doSave);
end

function phaseBands(ax, P, yl, labels)
% Shade every other mission phase (turn entry, turn and turn exit merged) and, optionally, name the phases.
names = P.phase;
names(startsWith(names, "turn")) = "turn";
names = replace(names, " 2", "");
% the +-0.3 m climb and descent is flown at cruise speed right after the second cruise (2 s): one "cruise" band, so
% that the short labels do not overlap
names(names == "climb/descent") = "cruise";
names(names == "final hover") = "hover";
groups = [1; find(names(2:end) ~= names(1:end-1)) + 1];
narrowW = 0.08 * (P.t_end(end) - P.t_start(1));      % a band narrower than this cannot hold its label alone
prevNarrowAbove = false;
for g = 1:numel(groups)
    a = P.t_start(groups(g));
    if g < numel(groups), b = P.t_start(groups(g + 1)); else, b = P.t_end(end); end
    if mod(g, 2) == 0
        patch(ax, [a b b a], [yl(1) yl(1) yl(2) yl(2)], [0.5 0.5 0.5], 'FaceAlpha', 0.08, 'EdgeColor', 'none', ...
              'HandleVisibility', 'off');
    end
    if labels
        % two narrow bands in a row (hover, transition): the second label goes inside the axes, under the top edge
        narrow = (b - a) < narrowW;
        inside = narrow && prevNarrowAbove;
        if inside, va = 'top'; else, va = 'bottom'; end
        text(ax, (a + b) / 2, yl(2), names(groups(g)), 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', va, 'FontSize', 7);
        prevNarrowAbove = narrow && ~inside;
    end
end
uistack(findobj(ax, 'Type', 'patch'), 'bottom');
end

function panel(S, E, fields, styles, scale, ylabs, figDir, name, doSave, refs)
% One panel file: n axes stacked on a common time axis, one quantity per axes, every variant in its colour and line
% style, legend of the variants on top. refs (optional): per axes, the field of a reference drawn black dotted ('' none).
n = numel(fields);
if nargin < 10, refs = repmat({''}, 1, n); end
f = newFigure(S, S.panelW, S.panelH);
tl = tiledlayout(f, n, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
href = gobjects(0);
for k = 1:n
    ax = nexttile(tl); hold(ax, 'on');
    if ~isempty(refs{k})
        href = plot(ax, E{1}.t, scale(k) * E{1}.(refs{k}), ':', 'Color', 'k', 'LineWidth', 1.1);
    end
    hk = overlay(ax, E, S, fields{k}, styles{k}, scale(k));
    if k == 1, h = hk; end
    timeAxis(ax, S, ylabs{k}, k == n, [0 12]);
end
if S.panelLegend
    names = S.labels;
    if ~isempty(href), h = [h, href]; names = [names, {'Reference'}]; end
    lgd = legend(h, names, 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 8);
    lgd.Layout.Tile = 'north';
end
saveFigure(f, figDir, name, doSave);
end

function pathPanel(S, E, figDir, name, doSave)
% 3D path of every variant and the reference (black dotted), at the size of the episode panels.
f = newFigure(S, S.panelW, S.panelH);
ax = axes(f); hold(ax, 'on');
set(ax, 'SortMethod', 'childorder');                  % reference on top in the -painters PDF (see figA1)
for i = 1:numel(E)
    plot3(ax, E{i}.x, E{i}.y, E{i}.z, S.lineStyle{i}, 'Color', S.color(i, :), 'LineWidth', 1.0);
end
plot3(ax, E{1}.x_ref, E{1}.y_ref, E{1}.z_ref, ':', 'Color', 'k', 'LineWidth', 1.1);
view(ax, -54, 12.6); daspect(ax, [1 1 0.25]);
xlabel(ax, '$x$ [m]'); ylabel(ax, '$y$ [m]'); zlabel(ax, '$z$ [m]');
formatAxis(ax, S);
saveFigure(f, figDir, name, doSave);
end

function legendOnDrawing(ax, lgd, cb)
% With a fixed data aspect ratio a 3D drawing is scaled down and centred inside the axes box, so a northoutside
% legend floats above a blank band. Place the legend just above the drawn plot box instead, centred over the drawing
% (and the colour bar cb, if given). The plot box is the projection of the corners of the axes limits, scaled by
% the data aspect ratio, fitted into the axes rectangle.
drawnow;
axPos = ax.Position;                                  % layout that already leaves room for the legend
lgd.Units = 'normalized';
lp = lgd.Position;
lgd.Location = 'none';
ax.Position = axPos;
[az, el] = view(ax);
T = viewmtx(az, el);
L = [xlim(ax); ylim(ax); zlim(ax)] ./ daspect(ax).';
[cx, cy, cz] = ndgrid(L(1, :), L(2, :), L(3, :));
uv = T(1:2, 1:3) * [cx(:) cy(:) cz(:)].';
f = ancestor(ax, 'figure');
fp = getpixelposition(f);
w = axPos(3) * fp(3); h = axPos(4) * fp(4);           % axes rectangle [px]
du = max(uv(1, :)) - min(uv(1, :)); dv = max(uv(2, :)) - min(uv(2, :));   % projected extent (no toolbox range)
s = min(w / du, h / dv);                              % scale of the fitted drawing
top = axPos(2) + (h + s * dv) / 2 / fp(4);            % top of the drawing, normalized to the figure
if isempty(cb)
    xMid = axPos(1) + axPos(3) / 2;
else
    cb.Units = 'normalized';
    xMid = (axPos(1) + cb.Position(1) + cb.Position(3)) / 2;
end
lgd.Position = [xMid - lp(3) / 2, min(top + 0.01, 1 - lp(4)), lp(3), lp(4)];
end

function legendFile(S, names, colors, styles, figDir, name, doSave)
% Legend only: invisible axes with one line per entry, horizontal, to place above the subfigures.
f = newFigure(S, 2 * S.panelW, 1.0);
ax = axes(f, 'Visible', 'off'); hold(ax, 'on');
h = gobjects(1, numel(names));
for k = 1:numel(names)
    h(k) = plot(ax, NaN, NaN, styles{k}, 'Color', colors{k}, 'LineWidth', 1.2);
end
lgd = legend(ax, h, names, 'Orientation', 'horizontal', 'NumColumns', numel(names), 'Box', 'off', ...
             'Interpreter', 'latex', 'FontSize', S.font);
lgd.Units = 'normalized'; lgd.Position(1:2) = [0.5 - lgd.Position(3) / 2, 0.5 - lgd.Position(4) / 2];
saveFigure(f, figDir, name, doSave);
end

function h = overlay(ax, E, S, fields, styles, scale) %#ok<INUSL>
% Every variant in its colour and line style (so that coinciding curves remain visible). Returns the handles of the
% first field.
h = gobjects(1, numel(E));
for i = 1:numel(E)
    for k = 1:numel(fields)
        hk = plot(ax, E{i}.t, scale * E{i}.(fields{k}), S.lineStyle{i}, 'Color', S.color(i, :), 'LineWidth', 1.2);
        if k == 1, h(i) = hk; end
    end
end
end

function timeAxis(ax, S, ylab, last, xl)
xlim(ax, xl); ylabel(ax, ylab); formatAxis(ax, S);
if last
    xlabel(ax, 'Time [s]');
else
    ax.XTickLabel = [];
end
end

function f = newFigure(S, widthCm, heightCm)
f = figure('Units', 'centimeters', 'Color', 'w', 'WindowStyle', 'normal');
f.Position = [2 2 widthCm heightCm];
set(f, 'DefaultAxesFontSize', S.font);
end

function formatAxis(ax, S)
set(ax, 'FontName', 'Computer Modern', 'FontSize', S.font, 'TickLabelInterpreter', 'latex', 'Box', 'on', 'TickDir', 'in', 'LineWidth', 0.6);
grid(ax, 'on'); ax.GridAlpha = 0.15;
end

function saveFigure(f, figDir, name, doSave)
% PNG with exportgraphics; PDF with print -painters (same path as the manual export of the figure window):
% the LaTeX text becomes vector outlines instead of selectable text set in a substitute font.
if ~doSave, return; end
drawnow;
if ~isvalid(f), warning('Figure %s was closed before it could be saved.', name); return; end
exportgraphics(f, fullfile(figDir, name + ".png"), 'Resolution', 300);
set(f, 'PaperUnits', 'centimeters');
pos = get(f, 'Position');
set(f, 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
print(f, fullfile(figDir, name + ".pdf"), '-dpdf', '-painters');
end
