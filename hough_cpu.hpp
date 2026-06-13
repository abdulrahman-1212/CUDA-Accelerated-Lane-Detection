#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <cstdint>

struct Line { int x1, y1, x2, y2; };

// Simple probabilistic Hough on CPU (edge image already small after region mask)
inline std::vector<Line> hough_transform_cpu(const uint8_t* edges,
                                              int width, int height,
                                              int threshold     = 20,
                                              int minLineLength = 20,
                                              int maxLineGap    = 500)
{
    // We replicate OpenCV HoughLinesP behaviour with a straightforward
    // accumulator approach — good enough for the region-masked edge image.
    const double rho_res   = 1.0;
    const double theta_res = M_PI / 180.0;
    const int    num_theta = 180;
    const int    diag      = (int)std::ceil(std::sqrt((double)width*width +
                                                       (double)height*height));
    const int    num_rho   = 2 * diag + 1;

    std::vector<int> accum(num_rho * num_theta, 0);

    std::vector<std::pair<int,int>> edge_pts;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if (edges[y * width + x] > 0)
                edge_pts.emplace_back(x, y);

    // Pre-compute cos/sin
    std::vector<double> cos_t(num_theta), sin_t(num_theta);
    for (int t = 0; t < num_theta; ++t) {
        double angle = t * theta_res;
        cos_t[t] = std::cos(angle);
        sin_t[t] = std::sin(angle);
    }

    for (auto [x, y] : edge_pts)
        for (int t = 0; t < num_theta; ++t) {
            int r = (int)std::round(x * cos_t[t] + y * sin_t[t]) + diag;
            if (r >= 0 && r < num_rho)
                ++accum[r * num_theta + t];
        }

    std::vector<Line> lines;
    for (int r_idx = 0; r_idx < num_rho; ++r_idx)
        for (int t = 0; t < num_theta; ++t) {
            if (accum[r_idx * num_theta + t] < threshold) continue;

            double rho   = (r_idx - diag) * rho_res;
            double theta = t * theta_res;
            double ct = cos_t[t], st = sin_t[t];

            // Walk along the line and collect segments
            int seg_x1 = -1, seg_y1 = -1, gap = 0;

            // Parametrise: for each point in the edge set close to this line
            // (fast approximation: just scan pixels on the line direction)
            for (auto [ex, ey] : edge_pts) {
                double dist = std::abs(ex * ct + ey * st - rho);
                if (dist > 1.5) { gap++; continue; }

                if (seg_x1 == -1) {
                    seg_x1 = ex; seg_y1 = ey;
                    gap = 0;
                } else if (gap > maxLineGap) {
                    int len = (int)std::hypot(ex - seg_x1, ey - seg_y1);
                    if (len >= minLineLength)
                        lines.push_back({seg_x1, seg_y1, ex, ey});
                    seg_x1 = ex; seg_y1 = ey;
                    gap = 0;
                } else {
                    gap = 0;
                }
            }
        }

    return lines;
}

struct LaneLine { double slope, intercept; };

inline std::optional<LaneLine> weighted_average(
    const std::vector<std::pair<LaneLine,double>>& weighted)
{
    if (weighted.empty()) return std::nullopt;
    double sw = 0, ss = 0, si = 0;
    for (auto& [ln, w] : weighted) { sw += w; ss += ln.slope * w; si += ln.intercept * w; }
    return LaneLine{ss/sw, si/sw};
}

struct LanePoints { int x1,y1,x2,y2; bool valid; };

inline LanePoints pixel_points(int img_height, const std::optional<LaneLine>& lane)
{
    if (!lane) return {0,0,0,0,false};
    double y1 = img_height;
    double y2 = img_height * 0.6;
    auto& [s, b] = *lane;
    if (std::abs(s) < 1e-6) return {0,0,0,0,false};
    int x1 = (int)((y1 - b) / s);
    int x2 = (int)((y2 - b) / s);
    return {x1, (int)y1, x2, (int)y2, true};
}

inline std::pair<LanePoints,LanePoints>
lane_lines_from_hough(const std::vector<Line>& lines, int img_height)
{
    std::vector<std::pair<LaneLine,double>> left_w, right_w;
    for (auto& l : lines) {
        if (l.x1 == l.x2) continue;
        double slope     = (double)(l.y2 - l.y1) / (l.x2 - l.x1);
        double intercept = l.y1 - slope * l.x1;
        double length    = std::hypot(l.y2 - l.y1, l.x2 - l.x1);
        if (slope < 0) left_w.push_back({{slope, intercept}, length});
        else           right_w.push_back({{slope, intercept}, length});
    }
    return { pixel_points(img_height, weighted_average(left_w)),
             pixel_points(img_height, weighted_average(right_w)) };
}
