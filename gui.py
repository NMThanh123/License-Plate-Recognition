from tkinter import *
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog, messagebox
from utility import *
from PIL import ImageTk, Image
import cv2 as cv


class Window(Toplevel):
    def __init__(self):
        super().__init__()

class GUI(Tk):
    idx = 0
    def __init__(self) -> None:
        # initialize gui for application
        super().__init__()
        self.resizable(0, 0)
        self.title('License plate recognition')
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry('%dx%d+%d+%d' % (1650, 680, (self.screen_width/2) - (1650/2), (self.screen_height/2) - (680/2)))
        self.iconbitmap('assets/icon.ico')
        self.style = Style()
        self.style.configure('TButton', foreground='blue', font=('arial', 15, 'bold'), borderwidth=0)
        self.style.configure('TLabel', font=('arial', 20, 'bold'))
        self.style.map('TButton', foreground=[('active','red')])
        
        self.label = Label(self, text='License Plate Recogniniton', style='TLabel')
        self.label.grid(row=0, column=0, columnspan=3,padx=(700,0), pady=(40,0), sticky='w')

        self.canv1 = Canvas(self, width=600, height=400, bg='white')
        self.canv1.grid(row=1, column=0, padx=(50,0), pady=(50,0))
        self.canv1.create_rectangle(1,1,603,403, outline= 'black', width=4, fill='white')

        self.img = ImageTk.PhotoImage(Image.open("assets/arrow.png").resize((250, 90)))
        self.arrow = Label(self, image=self.img)
        self.arrow.grid(row=1, column=1, pady=(50, 0), sticky='w')

        self.canv2 = Canvas(self, width=300, height=200)
        self.canv2.grid(row=1, column=2, padx=(10,0), pady=(50,0))

        self.entry = tk.Entry(width=11, justify=CENTER, background='white', foreground='black', font=('Helvetica', 25), highlightthickness=1, borderwidth=1, relief="solid")
        self.entry.place(relx=0.575, rely=0.58)

        self.btn1 = Button(self, text='Choose file', width=15, style='TButton', command=lambda: self.browser_img())
        self.btn1.grid(row=2, column=0,padx=(50, 0), pady=(20,0), sticky='w')

        self.btn2 = Button(self, text='Open camera', width=15, style='TButton', command=lambda: self.recognize_camera())
        self.btn2.grid(row=2, column=0,padx=(265,0), pady=(20,0), sticky='nw')

        self.btn3 = Button(self, text='Recoginize', width=15, style='TButton', command=lambda: self.recognize_one_image())
        self.btn3.grid(row=2, column=0,padx=(100, 0), pady=(20,0), sticky='e')

        self.btn4 = Button(self, text='Stop camera', width=15, style='TButton', command=lambda: self.stop_camera())
        self.btn4.grid(row=3, column=0,padx=(265,0), pady=(20,0), sticky='nw')

        self.btn5 = Button(self, text='Delete', width=10, style='TButton', command=lambda: self.delete_plate())
        self.btn5.grid(row=2, column=3,padx=(130,0), pady=(20,0), sticky='w')

        # add treeview
        columns = ('Oder', 'Number License Plate')
        self.tree = Treeview(self, columns=columns, height=19, show='headings')
        self.tree.grid(row=1, column=3, pady=(50, 0), sticky='w')

        # add a scrollbar in right treeview
        self.scrollbar = Scrollbar(self, orient=VERTICAL, command=self.tree.yview)
        self.scrollbar.grid(row=1, column=4, pady=(47, 0), sticky='ns')

        # initialize camera
        self.stop = 0
        self.cam = cv.VideoCapture(0)

        # initialize model
        self.det, self.rec = initialize_model()

        # show plate in treeview
        self.show_plate()

    
    def browser_img(self):
        """
        function to insert image from computer
        """
        global yourImage, img
        self.stop=1
        yourImage=filedialog.askopenfilename(title = "Select your image", filetypes = [("Image Files","*.png"),("Image Files","*.jpg")])
        imgfile=Image.open(yourImage).resize((600, 400))
        imgToInsert=ImageTk.PhotoImage(imgfile)
        self.canv1.image = imgToInsert
        img = self.canv1.create_image(1.5, 1.5, anchor=NW, image=imgToInsert, tags='img')

    def recognize(self, image):
        """
        recognize number plate and insert characters into treeview
        """
        number_plate = [None]
        image = cv.resize(image, (640, 640))
        image_recognize = process_image(image)
        boxes = self.det.run(None, {'images': image_recognize})
        for bbox in boxes:
            if np.max(bbox[0][4]) > 0.8:
                conf = bbox[0][4]
                max_conf = np.where(conf == np.max(conf))[0][0]
                box = bbox[0,:,max_conf][:4]
                x1, y1, x2, y2 = get_box(box)
                if self.stop == 0:
                    if x1 > 140 and 140 < x2 < 450 and y1 > 140 and  140 < y2 < 450:
                        plate = image[int(y1)-1: int(y2)+1, int(x1)-3: int(x2)+3]
                        plate = adjust_image(unwrap_image(plate))
                        plate = self.resize_image(plate)
                        platetk = ImageTk.PhotoImage(image=Image.fromarray(plate))
                        self.canv2.image = platetk
                        self.canv2.create_image(self.canv2.winfo_width()//2-20, self.canv2.winfo_height()//2, anchor=CENTER, image=platetk)
                        number_plate = self.rec.ocr(plate, det=True, cls=False)
                else:
                    plate = image[int(y1)-1: int(y2)+1, int(x1)-3: int(x2)+3]
                    plate = adjust_image(unwrap_image(plate))
                    plate = self.resize_image(plate)
                    platetk = ImageTk.PhotoImage(image=Image.fromarray(plate))
                    self.canv2.image = platetk
                    self.canv2.create_image(self.canv2.winfo_width()//2-20, self.canv2.winfo_height()//2, anchor=CENTER, image=platetk)
                    number_plate = self.rec.ocr(plate, det=True, cls=False)
                if number_plate == [None]:
                    pass
                else :
                    txts = [line[1][0] for line in number_plate[0]]
                    text = format_string(''.join(t for t in txts))
                    if len(text) < 15:
                        check = 0
                        for number_plate in self.get_plate():
                            if check_plate(text, number_plate) > 95:
                                check = 1
                        if not check:
                            self.entry.delete(0, END)
                            self.entry.insert(0, text)
                            self.idx += 1
                            self.tree.insert('', END, values=[self.idx, f'{text}'])
                                
    
    def get_plate(self):
        """
        get plate in treeview
        """
        plates = []
        for item in self.tree.get_children():
            plate = self.tree.item(item)['values'][1]
            plates.append(plate)
        return plates


    def recognize_one_image(self):
        """
        function to recognize license number plate from image
        input: image
        output: plate and characters 
        """
        image = cv.imread(yourImage)
        self.recognize(image)

    
    def recognize_camera(self):
        """
        open camera and auto recognize number license plate
        input: image from camere
        output: plate and characters
        """
        global img_camera
        if self.stop == 1:
            if len(self.canv1.gettags('img1')) == 0:
                pass
            else:
                self.canv1.delete(img_camera)
            self.cam.release()
            self.cam = cv.VideoCapture(0)
            self.stop = 0
        
        else:
            _, frame = self.cam.read()
            cv.rectangle(frame, (140, 120), (470, 350), color=(0,255,0), thickness=2)
            self.recognize(frame)
            img_camera = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2RGBA) , (600, 400))
            img_camera = ImageTk.PhotoImage(image=Image.fromarray(img_camera))
            self.canv1.image = img_camera
            img_camera = self.canv1.create_image(1.5, 1.5, anchor=NW, image=img_camera, tags='img1')
            self.canv1.after(5, self.recognize_camera)

    
    def stop_camera(self):
        """
        function to stop camera or clear screen 
        """
        global img, img_camera
        self.stop = 1
        tag_camera = self.canv1.gettags('img1')
        tag = self.canv1.gettags('img')
        if len(tag_camera) == 0 and len(tag) == 0:
            pass
        if len(tag_camera) > 0:
            self.canv1.delete(img_camera)
        if len(tag) > 0:
            self.canv1.delete(img)

    @staticmethod
    def resize_image(img):
        """
        function to resize number licnse plate to show --> function recognize()
        """
        h, w = img.shape[:2]
        if h > 200:
            h = 140
        if w > 250:
            w = 200
        plate = cv.resize(img, (w, h))
        return plate
    
    def show_plate(self):
        """
        function to show number license plate in treeview
        """
        self.tree.column('Oder', anchor=CENTER, width=100)
        self.tree.heading('Oder', text='Oder')
        self.tree.column('Number License Plate', anchor=CENTER, width=250)
        self.tree.heading('Number License Plate', text='Number License Plate')
        self.tree.configure(yscroll=self.scrollbar.set)
      

    def delete_plate(self):
        """
        function for delete button(delete one or all of plate)
        """
        if len(self.tree.get_children()) == 0:
            messagebox.showinfo('Infor', 'No license number plate in treeview.')
            return False
        
        else:
            def choice(option):
                if option == 'all':
                    self.tree.delete(*self.tree.get_children())
                    self.idx = 0
                    pop.destroy()
                elif option == 'one':
                    if len(self.tree.selection()) == 1:
                        selected_item = self.tree.selection()[0]
                        self.tree.delete(selected_item)
                        for i, item in enumerate(self.tree.get_children()):
                            self.tree.item(item, values=(i+1, self.tree.item(item)['values'][1]))
                        self.idx -= 1
                        pop.destroy()
                        
                    else:
                        messagebox.showerror('Error', 'Please choose one to delete.')

        pop = Window()
        pop.title('Confirm delete?')
        pop.geometry('550x100')
        pop.resizable(0, 0)
        label = Label(pop, text="Would you like to delete all or selected number plate?", font=('Helvetica', 15, 'bold'))
        label.grid(row=0, column=0, padx=(30, 0), pady=(10,0))
        button1 = Button(pop, text="Delete all", command=lambda: choice("all"))
        button1.place(relx=0.45, rely=0.5)
        button2 = Button(pop, text="Delete one", command=lambda: choice("one"))
        button2.place(relx=0.73, rely=0.5)

if __name__ == '__main__':
    app = GUI()
    app.mainloop()


