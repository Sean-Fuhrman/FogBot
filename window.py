import pygame
background_color = (232, 162, 232)
screen = pygame.display.set_mode((300, 300)) ## pixel dimensions
pygame.display.set_caption('Fog of War Chess Board')
screen.fill(background_color) ## fills screen with background color
pygame.display.flip()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
