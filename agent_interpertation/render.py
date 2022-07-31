import json
import math

import matplotlib
import numpy as np
import pygame
import torch
import cv2


X = 1
Y = 0


def dist_from(agent, others):
    return torch.linalg.norm(agent - others,dim=-1)


def clear_patches(ax):
    [p.remove() for p in reversed(ax.patches)]
    [t.remove() for t in reversed(ax.texts)]


def get_desired_pos_vector(position, target_position, desired_dist_target):
    """
    R: desired distance to target
    p_t: target position
    p: base position
    q = p - p_t
    U(q, R) = (||q||_2 - R)^2
    d/dq U(q, R) = q * (2 - (2 * R)/||q||_2)
    """
    q = np.array(target_position) - np.array(position)
    v = q * (2 - (2 * desired_dist_target / np.sqrt(np.sum(q ** 2))))
    return v  # / np.linalg.norm(v)


GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (237, 183, 57)
RED = (235, 64, 52)
BROWN = (79, 58, 9)
confidence_thickness = 10

CONF_BOUNDARY = 2


def render_just_targets(targets,color,conf_color,lowconfhue,display,whichtargets,factor=10, config_file="../../data/params.json",):
    lowconfhue = (lowconfhue,)
    with open(config_file, "r") as p:
        cfg = json.load(p)["env_config"]
        cfg["render_px_per_m"] *= factor

    dim = np.array(np.max(cfg["world_perimeter"], axis=0) - np.min(cfg["world_perimeter"], axis=0))

    def point_to_screen(p):
        return ((np.array(p) + dim / 2) * cfg["render_px_per_m"]).astype(float)

    pygame.init()  # now use display


    target_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)
    for i in whichtargets:
        target = targets[i]
        pos = target[:, 0]
        pos_confidence = target[:, 1] * 2
        if torch.max(pos_confidence) > CONF_BOUNDARY:
            col = color + lowconfhue
        else:
            col = color
        pygame.draw.circle(
            target_img,
            col,
            point_to_screen(pos),
            cfg["agent_radius"] / 2 * cfg["render_px_per_m"])

        if torch.max(pos_confidence) < 10:
            if torch.max(pos_confidence) > CONF_BOUNDARY:
                col = conf_color + lowconfhue
            else:
                col = conf_color
            pos_rectangle = pygame.Rect(point_to_screen(pos - pos_confidence),
                                        (pos_confidence * 2 * cfg["render_px_per_m"]).tolist())

            pygame.draw.ellipse(
                target_img,
                col,
                pos_rectangle, width=confidence_thickness)
        else:
            print("STD too high for pos")
    display.blit(target_img, (0, 0))

    return display
def render(robots, targets, factor=10, config_file="../../data/params.json", savefile=None, mode=None, display=None,
           zs=None, message=None, zmin=None, zmax=None):
    with open(config_file, "r") as p:
        cfg = json.load(p)["env_config"]
        cfg["render_px_per_m"] *= factor

    dim = np.array(np.max(cfg["world_perimeter"], axis=0) - np.min(cfg["world_perimeter"], axis=0))

    def point_to_screen(p):
        return ((np.array(p) + dim / 2) * cfg["render_px_per_m"]).astype(float)

    size = (dim * cfg["render_px_per_m"]).astype(int)
    if display is None:
        display = pygame.display.set_mode(size, pygame.HIDDEN)

    AGENT_COLOR = BLUE
    BACKGROUND_COLOR = WHITE
    TARGET_COLOR = GREEN
    PERIMETER_COLOR = WHITE

    display.fill(WHITE)

    pygame.init()  # now use display
    font = pygame.font.SysFont("latinmodernsans", 30 * factor)

    if zs is not None:
        zs_image = pygame.Surface(display.get_size(), pygame.SRCALPHA)
        cmap = matplotlib.cm.get_cmap('viridis')

        def gcol(val):
            return [x * 255 for x in cmap(norm(val))[:-1]]

        z_box_size = cfg["agent_radius"] * 2
        for i, z_val in enumerate(zs):
            norm = matplotlib.colors.Normalize(vmin=zmin[i], vmax=zmax[i])
            z_val = z_val[0]
            rect = pygame.Rect((0, size[1] - z_box_size * (i + 1) * cfg["render_px_per_m"]),
                               (z_box_size * cfg["render_px_per_m"], z_box_size * cfg["render_px_per_m"]))
            pygame.draw.rect(zs_image, gcol(z_val), rect)
        display.blit(zs_image, (0, 0))

    if message is not None:
        message_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)
        norm = matplotlib.colors.Normalize(vmin=0.5, vmax=0.8)
        cmap = matplotlib.cm.get_cmap('viridis')

        def gcol(val):
            return [x * 255 for x in cmap(norm(val))[:-1]]

        message_box_size = cfg["target_radius"] * 2
        for i, m_val in enumerate(message):
            m_val = m_val.item()
            rect = pygame.Rect((size[0] - message_box_size * (i + 1) * cfg["render_px_per_m"], 0),
                               (message_box_size * cfg["render_px_per_m"], message_box_size * cfg["render_px_per_m"]))
            pygame.draw.rect(message_img, gcol(m_val), rect)
        display.blit(message_img, (0, 0))

    target_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)
    for target in targets:
        for robot in robots:
            if (
                    np.linalg.norm(robot - target)
                    > cfg["visibility_range"]
            ):
                continue

            pygame.draw.line(
                target_img,
                TARGET_COLOR,
                point_to_screen(robot),
                point_to_screen(target),
                4 * factor,
            )

        target_color = TARGET_COLOR
        pygame.draw.circle(
            target_img,
            target_color,
            point_to_screen(target),
            cfg["target_radius"] * cfg["render_px_per_m"],
        )
    display.blit(target_img, (0, 0))
    for i, robot in enumerate(robots):
        robot_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)

        pygame.draw.circle(
            robot_img,
            PERIMETER_COLOR + (40,),
            point_to_screen(robot),
            cfg["perimeter_visibility_range"] * cfg["render_px_per_m"],
            width=2,
        )
        pygame.draw.circle(
            robot_img,
            AGENT_COLOR + (50,),
            point_to_screen(robot),
            cfg["visibility_range"] * cfg["render_px_per_m"],
        )
        pygame.draw.circle(
            robot_img,
            AGENT_COLOR,
            point_to_screen(robot),
            cfg["agent_radius"] * cfg["render_px_per_m"],
        )
        robot_label = font.render(f"{i}", False, (0, 0, 0))
        robot_label = pygame.transform.rotate(robot_label, 90)
        display.blit(
            robot_label,
            point_to_screen(robot + cfg["agent_radius"] / 2),
        )

        for other_robot in robots:
            if (
                    np.linalg.norm(robot - other_robot)
                    > cfg["visibility_range"]
            ):
                continue

            pygame.draw.line(
                robot_img,
                AGENT_COLOR,
                point_to_screen(robot),
                point_to_screen(other_robot),
                4 * factor,
            )

        display.blit(robot_img, (0, 0))

    perimeter_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)
    for perimeter_pose in cfg["world_perimeter"]:
        for robot in robots:
            if (
                    np.linalg.norm(robot - torch.tensor(perimeter_pose))
                    > cfg["perimeter_visibility_range"]
            ):
                continue

            pygame.draw.line(
                perimeter_img,
                PERIMETER_COLOR,
                point_to_screen(robot),
                point_to_screen(perimeter_pose),
                4 * factor,
            )

        pygame.draw.circle(
            perimeter_img,
            PERIMETER_COLOR,
            point_to_screen(perimeter_pose),
            0.1 * cfg["render_px_per_m"],
        )
    display.blit(perimeter_img, (0, 0))

    if savefile is not None:
        pygame.display.update()
        pygame.image.save(display, savefile)
        image = cv2.imread(savefile)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(savefile, image)
    if mode == "human":
        pygame.display.update()
    elif mode == "rgb_array":
        pygame.display.update()
        return pygame.surfarray.array3d(display), display
    elif mode == "build":
        return display, cfg, dim


PURPLE = (244, 3, 252)


def render_on_perception(display, pos_mean_std, obs_agents_mean_std, obs_targets_mean_std, cfg, dim,
                         target_predictions=None, config_file="../../data/params.json", return_multiple=False,
                         factor=10,
                         agent_color=PURPLE,target_color=GREEN,lowconfhue=100):
    lowconfhue = (lowconfhue,)
    POS_CONF = (0, 0, 0)
    TARGET_COLOR = (252, 161, 3)
    OTHERS_COLOR = (71, 162, 201)
    TARGET_CONF = (79, 67, 15)
    OTHER_CONF = (22, 76, 99)

    def point_to_screen(p):
        return ((np.array(p) + dim / 2) * cfg["render_px_per_m"]).astype(float)

    if display is None:
        pygame.init()
        with open(config_file, "r") as p:
            cfg = json.load(p)["env_config"]
            cfg["render_px_per_m"] *= factor
            dim = np.array(np.max(cfg["world_perimeter"], axis=0) - np.min(cfg["world_perimeter"], axis=0))
        display = pygame.display.set_mode((dim * cfg["render_px_per_m"]).astype(int), pygame.HIDDEN)
        display.fill(WHITE)

    if target_predictions is not None:

        TARGET_COLOR = target_color
        TARGET_CONF = (0, 100, 0)
        target_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)

        for target in target_predictions:
            tpos = target[:, 0]
            tpos_confidence = target[:, 1] * 2
            if torch.max(tpos_confidence)>1:
                col = TARGET_COLOR + lowconfhue
            else:
                col = TARGET_COLOR
            pygame.draw.circle(
                target_img,
                col,
                point_to_screen(tpos),
                cfg["target_radius"] * cfg["render_px_per_m"],
            )

            if torch.max(tpos_confidence) < 10:
                if torch.max(tpos_confidence) > CONF_BOUNDARY:
                    col = TARGET_CONF + lowconfhue
                else:
                    col = TARGET_CONF
                tpos_rectangle = pygame.Rect(point_to_screen(tpos - tpos_confidence),
                                             (tpos_confidence * 2 * cfg["render_px_per_m"]).tolist())

                pygame.draw.ellipse(
                    target_img,
                    col,
                    tpos_rectangle, width=confidence_thickness)
            else:
                print("STD too high for target_prediction")

    robot_img = pygame.Surface(display.get_size(), pygame.SRCALPHA)

    pos = pos_mean_std[:, 0]
    pos_confidence = pos_mean_std[:, 1] * 2
    if torch.max(pos_confidence) > CONF_BOUNDARY:
        col = agent_color + lowconfhue
    else:
        col = agent_color
    pygame.draw.circle(
        robot_img,
        col,
        point_to_screen(pos),
        cfg["agent_radius"] / 2 * cfg["render_px_per_m"])

    if torch.max(pos_confidence) < 10:
        if torch.max(pos_confidence) > CONF_BOUNDARY:
            col = POS_CONF + lowconfhue
        else:
            col = POS_CONF
        pos_rectangle = pygame.Rect(point_to_screen(pos - pos_confidence),
                                    (pos_confidence * 2 * cfg["render_px_per_m"]).tolist())

        pygame.draw.ellipse(
            robot_img,
            col,
            pos_rectangle, width=confidence_thickness)
    else:
        print("STD too high for pos")

    def draw_ellipse_angle(surface, color, center, width, height, angle, thickness=0, radians=True):
        if radians:
            angle = angle * 180 / np.pi
        target_rect = pygame.Rect(point_to_screen((center[0].item() - width / 2, center[1].item() - height / 2)),
                                  (width * cfg["render_px_per_m"], height * cfg["render_px_per_m"]))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), thickness)
        rotated_surf = pygame.transform.rotate(shape_surf, angle)
        surface.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))

    def draw_angled_trapezpid(surface, color, center, center_dist, conf, trapez_angle, angle, thickness=0,
                              radians=True):
        if radians:
            angle = angle * 180 / np.pi

        if center_dist < conf:
            near = center_dist
        else:
            near = conf
        br = (math.tan(trapez_angle) * (center_dist - near), -conf)
        bl = (- math.tan(trapez_angle) * (center_dist - near), -conf)
        tr = (math.tan(trapez_angle) * (center_dist + conf), conf)
        tl = (- math.tan(trapez_angle) * (center_dist + conf), conf)

        def mult_move_all(points):
            for i in range(len(points)):
                points[i] = (
                (-tl[0] + points[i][0]) * cfg["render_px_per_m"], (-bl[1] + points[i][1]) * cfg["render_px_per_m"])
            return points

        total_tl = point_to_screen((center[0].item() + tl[0], center[1].item() - conf))
        target_rect = pygame.Rect(total_tl,
                                  (round(cfg["render_px_per_m"] * (tr[0] - tl[0])),
                                   round(cfg["render_px_per_m"] * (tl[1] - bl[1]))))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surf, color, mult_move_all([br, bl, tl, tr]), thickness)
        rotated_surf = pygame.transform.rotate(shape_surf, angle)
        surface.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))

    num_angles = obs_targets_mean_std.shape[0]
    angle_increment = 2 * np.pi / num_angles

    if obs_agents_mean_std is not None:
        for i, angle in enumerate(
                np.linspace(0 + angle_increment / 2, 2 * np.pi + angle_increment / 2, num_angles, endpoint=False)):
            dist = (1 - obs_agents_mean_std[i][0].item()) * cfg["visibility_range"]
            dist = max(dist, 0.01)
            std = obs_agents_mean_std[i][1].item()
            if std < 5:
                if std>1:
                    col = OTHERS_COLOR + lowconfhue
                else:
                    col = OTHERS_COLOR
                if dist > 0 and dist < cfg["visibility_range"]:
                    other_pos = pos + np.asarray([
                        dist * math.cos(angle),
                        dist * math.sin(angle),
                    ])
                    pygame.draw.circle(
                        robot_img,
                        col,
                        point_to_screen(other_pos),
                        cfg["agent_radius"] / 2 * cfg["render_px_per_m"])
                    if std > CONF_BOUNDARY:
                        col = OTHER_CONF + lowconfhue
                    else:
                        col = OTHER_CONF
                    # width = 2 * cfg["agent_radius"]
                    # height = 4 * std
                    # draw_ellipse_angle(robot_img, ABLACK, other_pos, width, height, angle=np.pi / 2 - angle, thickness=40)
                    draw_angled_trapezpid(robot_img, col, other_pos, dist, 2 * std, angle_increment,
                                          np.pi / 2 - angle,
                                          thickness=confidence_thickness)
            else:
                print("STD too high for obs_agent: ",std)

    for i, angle in enumerate(
            np.linspace(0 + angle_increment / 2, 2 * np.pi + angle_increment / 2, num_angles, endpoint=False)):
        dist = (1 - obs_targets_mean_std[i][0].item()) * cfg["visibility_range"]
        std = obs_targets_mean_std[i][1].item()
        if std < 5:
            dist = max(dist, 0.01)
            if std > CONF_BOUNDARY:
                col = TARGET_COLOR + lowconfhue
            else:
                col = TARGET_COLOR
            if dist > 0 and dist < cfg["visibility_range"]:
                other_pos = pos + np.asarray([
                    dist * math.cos(angle),
                    dist * math.sin(angle),
                ])
                pygame.draw.circle(
                    robot_img,
                    col,
                    point_to_screen(other_pos),
                    cfg["target_radius"] * cfg["render_px_per_m"])

                if std > CONF_BOUNDARY:
                    col = TARGET_CONF + lowconfhue
                else:
                    col = TARGET_CONF
                # width = 2 * cfg["agent_radius"]
                # height = 4 * std
                # draw_ellipse_angle(robot_img, ABLACK, other_pos, width, height, angle=np.pi / 2 - angle, thickness=confidence_thickness)
                draw_angled_trapezpid(robot_img, col, other_pos, dist, 2 * std, angle_increment,
                                      np.pi / 2 - angle,
                                      thickness=4)
        else:
            print("STD too high for obs_target")
    display.blit(robot_img, (0, 0))
    if return_multiple:
        return display, cfg, dim
    return display