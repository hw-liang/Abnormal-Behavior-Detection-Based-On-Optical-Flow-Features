import lxml.etree as et

class Poi_handle():

    def __init__(self,path='../ref_data/poi.xml'):
        parser = et.XMLParser(remove_blank_text=True)
        self.tree = et.parse(path,parser)

    def searchPic(self,n):
        for pic in self.tree.getroot():
            if pic.get('n') == str(n).zfill(3):
                return pic

    def add(self,n,y,*xs):
        picPt=None
        for subEl1 in self.tree.getroot():
            if subEl1.get('n') == str(n):
                picPt=subEl1;
                break

        yPt=None
        if picPt is not None:
            for subEl2 in picPt:
                if subEl2.get('val') == str(y):
                    yPt = subEl2;
                    break
            if yPt is None:
                yPt=et.SubElement(picPt,'y',val=str(y))
        else:
            yPt=et.SubElement(et.SubElement(self.tree.getroot(), "pic", n = str(n) ),'y',val=str(y))

        for child in yPt:
            yPt.remove(child)
        for x in xs:
            et.SubElement(yPt, "point").text=str((x,y))
        self.tree.write('../ref_data/poi.xml',pretty_print=True)



