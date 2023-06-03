import cv2

# pour sélectionner une partie d'une vidéo
# appuyer sur a pour commencer la sélection, sur b pour terminer la sélection
def process_video(path, name, start=0):
    out = cv2.VideoWriter(path + name + '_processed.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (450, 800))
    cap = cv2.VideoCapture(path + name + '.mp4')
    
    saving = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for _ in range(start, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) :
        photo = cap.read()[1]           # Storing the frame in a variable photo

        # éventuellement crop:
        #h, w, _ = photo.shape
        #nh = w * 4 / 3
        #mid = h / 2
        #photo = photo[int(mid - nh / 2):int(mid + nh / 2), :]
        #photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
        #photo = cv2.resize(photo, (450, 800))

        photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE) # jsp pk mais les vidéos sont pas dans le bon sens par défaut
        if saving: out.write(photo)

        cv2.imshow("pspsps", photo)
        k = cv2.waitKey(0)

        if k == ord('a'):
            saving = True
        elif k == ord('b'):
            out.release()
            return
    out.release()

#process_video('21_03_2023/', 'v1_raw', 120*3)
process_video('04_04_2023/', 'v1_raw')