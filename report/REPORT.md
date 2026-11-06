 # Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Thông tin record hạng 1 của ảnh traffic:
class_id: 468
class_name: cab (xe taxi)
rank: 1
score: 0.510915 (51.09%)
taxonomy_name: ImageNet-1K

2. Đối tượng của dự đoán (Cả bức ảnh hay vật thể riêng biệt):
Record này nói về cả bức ảnh.
Lý do: Đây là tác vụ Phân loại ảnh (Image Classification). Mô hình gán nhãn đại diện cho toàn bộ nội dung hoặc chủ thể nổi bật nhất bao quát cả bức ảnh chứ không xác định vị trí, khoanh vùng hay bóc tách từng vật thể riêng lẻ.

3. Nguồn gốc của danh sách tên lớp (Class list):
Danh sách tên lớp đi kèm với taxonomy/checkpoint (cụ thể là từ bộ dữ liệu tiêu chuẩn ImageNet-1K được tích hợp sẵn khi huấn luyện mô hình). Mô hình không tự nghĩ ra tên lớp mà chỉ tính toán phân phối xác suất trên danh sách cố định này.

4. Lý do cần ghi đủ class_id, class_name và taxonomy_name:
    - class_id: Tránh nhầm lẫn khi lập trình, lưu trữ dữ liệu hoặc khi dịch thuật (các hệ thống khác nhau có thể dịch tên lớp khác nhau nhưng ID dạng số chỉ nhất quán trong cùng taxonomy và phiên bản ánh xạ lớp).
    - class_name: Giúp con người (annotator, reviewer, developer) dễ dàng đọc hiểu nhãn trực quan.
    - taxonomy_name: Để phân biệt ngữ cảnh của bộ nhãn. Ví dụ, lớp "cab" trong ImageNet-1K có thể có mã số và định nghĩa phạm vi khác so với lớp tương tự trong COCO hay OpenImages.

5. Quy định trong Guideline khi ảnh có nhiều vật / chủ đề:
    Khi ảnh có nhiều vật thể hoặc chủ đề, annotator phải chọn nhãn theo guideline và taxonomy của dự án. Guideline cần quy định rõ tiêu chí chọn chủ thể chính (chẳng hạn độ nổi bật hoặc vị trí nếu dự án yêu cầu), cách xử lý nhiều vật cùng mức độ nổi bật và trường hợp không có lớp phù hợp. Không tự tạo lớp ngoài taxonomy; nếu chưa có quy tắc rõ ràng thì escalation để thống nhất.

6. Vì sao không lấy score của mô hình làm ground truth:
    - Score là điểm tin cậy do mô hình tính toán, không phải bằng chứng rằng dự đoán đúng và cũng không phải điểm chất lượng nhãn người.
    - Mô hình có thể dự đoán sai dù score cao. Ground truth là nhãn tham chiếu được gán và kiểm tra theo guideline; nhãn này vẫn có thể có sai sót nên cần quy trình reviewer/QC.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Trích xuất một record dự đoán (sample_id = kitchen):
Chúng ta chọn đối tượng người đứng ở trung tâm ảnh làm ví dụ:
class_name: person
score: 0.912625
bbox_xyxy: [385.33, 69.24, 498.92, 348.92] (tọa độ pixel)
bbox_width: 113.58
bbox_height: 279.68

2. Giải thích tọa độ bounding box bằng lời thường:
Vị trí góc trên-trái: Nằm tại tọa độ pixel x = 385.33 (cách mép trái ảnh khoảng 385 pixel) và y = 69.24 (cách mép trên ảnh khoảng 69 pixel).
Vị trí góc dưới-phải: Nằm tại tọa độ pixel x = 498.92 và y = 348.92.
Kích thước: Bounding box bao quanh người này có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.

3. Số lượng dự đoán (prediction) theo các ngưỡng khác nhau:
Theo số liệu đã ghi từ ô so sánh threshold trong notebook (cần đối chiếu lại ô output trước khi nộp vì JSON đính kèm chỉ được xuất ở ngưỡng 0.35):
- Ngưỡng 0.20: 17 dự đoán (chưa xác minh độc lập).
- Ngưỡng 0.60: 6 dự đoán (chưa xác minh độc lập).
4. Công việc của Reviewer khi hạ thấp ngưỡng lọc (Threshold):
Khi hạ ngưỡng xuống thấp (giữ lại nhiều hộp dự đoán hơn):
    - Reviewer phải mất nhiều thời gian hơn để rà soát vì xuất hiện thêm nhiều hộp nhiễu (False Positives).Họ phải thực hiện thêm các thao tác xóa bỏ (delete) các hộp đoán sai hoàn toàn của mô hình, hoặc sửa đổi nhãn bị mô hình gán nhãn nhầm do độ tin cậy thấp.
5. Quy tắc vẽ bounding box chung (Guideline):
Quy tắc: "Vẽ bounding box sát phần vật thể nhìn thấy được, hạn chế chứa nền thừa và không cắt mất phần vật thể quan sát được. Với vật bị che khuất hoặc cắt mép ảnh, áp dụng đúng quy tắc occlusion/truncation của guideline; nếu chưa được quy định thì escalation."
6. Trường hợp vật bị che khuất hoặc cắt mép trên ảnh kitchen:
-Vật thể quan sát được: Chiếc chảo / nồi treo ở phía trên bên trái bị che khuất một phần hoặc các đĩa bánh ở trên bàn bị che khuất một phần bởi cây lăn bột.
Hướng giải quyết: Đây là trường hợp cần đưa lên hỏi Lab Coach (escalation).
Lý do: Những trường hợp bị che khuất phức tạp (occlusion) hoặc bị cắt rìa ảnh nghiêm trọng rất dễ gây ra sự thiếu nhất quán giữa các annotators. Việc hỏi Lab Coach giúp đội ngũ quản lý thống nhất một quy chuẩn chung (ví dụ: chỉ gán nhãn nếu phần nhìn thấy được lớn hơn 30%), từ đó viết bổ sung rõ ràng vào Guideline chính thức thay vì để mọi người tự gán nhãn theo cảm tính cá nhân.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-001` ; `class_name`=`person` (`class_id`=0, COCO-80) ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=`[446, 70]`, `[445, 71]`, `[444, 71]`, `[443, 72]`, `[442, 72]` (đơn vị pixel).
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ cho biết vật nằm trong một hình chữ nhật nên bao luôn cả nền (khoảng trống giữa hai chân, hai bên tay, góc quanh đầu). Polygon 348 điểm đi theo đường viền thật của người: tròn theo đầu, nhô ra ở hai khuỷu tay chống hông, và có chỗ lõm giữa hai chân (khoảng điểm `[433, 251]` → `[435, 252]`). Polygon mô tả đường biên chi tiết hơn box và có thể chuyển thành mask để ước lượng vùng pixel, diện tích, hình dạng của vật. Tuy nhiên, đây vẫn là prediction của mô hình, không bảo đảm đường biên trùng hoàn toàn với vật thể thật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` (vd. `kitchen-001`) chỉ là số thứ tự để phân biệt từng đối tượng riêng lẻ trong **một ảnh**, nhất là khi nhiều vật cùng lớp (ảnh kitchen có 4 `bowl`: `kitchen-002`, `-003`, `-006`, `-010`, mỗi cái có polygon riêng).
    - Không phải class ID: class ID (`class_id` = 0 là person, 45 là bowl) nói vật thuộc *loại* nào, còn `instance_id` nói đó là *cái nào* trong ảnh.
    - Không phải tracking ID: tracking ID nối cùng một vật qua nhiều khung hình video. `instance_id` chỉ được đánh lại trong từng ảnh, nên `kitchen-001` không có ý nghĩa ở ảnh khác.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc (đối với **visible instance segmentation**): "Vẽ polygon bám sát đường biên phần vật thể thực sự nhìn thấy được, tránh lấy nhầm pixel nền hoặc vật khác; không đưa phần bị che khuất vào visible mask. Nếu dự án yêu cầu amodal mask thì làm theo guideline riêng."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Chỗ khó: chiếc bàn gỗ (`kitchen-005`, dining table 0.58) chạm và chồng lên các bát (`kitchen-003`, `kitchen-006`), phần bột và lò nướng (`kitchen-007`) ở góc trái. Mask bàn có tới 598 điểm, đường biên nhảy qua lại và lẫn cả vùng bát lẫn bột; mask lò nướng cũng chồng lên mặt bàn.
Dùng **guideline** về visible mask để kiểm tra ranh giới giữa bàn và các vật nằm phía trước. Nếu mask bàn chứa pixel của bát hoặc vật khác thì cần sửa theo quy tắc; không kết luận mọi mask đều tuyệt đối không được chồng nhau trong mọi kiểu annotation. Nếu guideline không đủ rõ thì escalation.
Riêng "person 0.43" ở mép trái (thực chất chỉ thấy bàn tay/cánh tay) thì **escalation** hỏi Lab Coach: có gán `person` cho vật chỉ thấy một phần nhỏ không? Hỏi để guideline ghi rõ, không tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một nhãn tham chiếu cho cả ảnh** theo guideline: `class_id` + `class_name` + `taxonomy_name`; 468 / cab / ImageNet-1K chỉ là **prediction top-1**, chưa được xác nhận là ground truth. | Ảnh `traffic` có rất nhiều bus và xe con nhưng top-1 là `cab` chỉ 0.51, kế đó `minibus` 0.16. Ảnh `kitchen` top-1 là `gong` 0.42, chưa chắc phản ánh đúng nội dung ảnh; việc taxonomy không có lớp `kitchen` không tự chứng minh dự đoán `gong` sai. Chủ đề không rõ một vật chính. | Chọn một lớp trong taxonomy theo tiêu chí đã được guideline quy định, không mặc định lấy vật lớn nhất/trung tâm. Không có lớp phù hợp thì escalation, không tự bịa tên lớp. | Nhãn có đúng nội dung ảnh và quy tắc không, có nằm trong taxonomy không, có nhất quán giữa các ảnh tương tự không. Không dùng score của model làm đáp án. |
| Phát hiện vật thể | **Một box + một lớp cho mỗi vật**: `bbox_xyxy` theo pixel + `class_name`. | `traffic`: bus ở xa rất nhỏ (~38×40 px, score 0.53–0.59), người bị cắt mép trái (`x` = 0.08). `kitchen`: box bowl 0.38 `[0.32, 305.02, 87.69, 371.32]` chồng lên box bowl 0.50 `[59.29, 288.92, 133.46, 328.98]` (cần kiểm tra ảnh trước khi coi là lỗi); "person 0.61" ở mép trái chỉ là bàn tay. | Vẽ box sát mép vật nhìn thấy, gán đúng lớp, theo quy tắc về vật nhỏ/bị che/bị cắt mép của guideline; ca mơ hồ thì escalation. | Box có sát không, có sót hoặc thừa vật nào không (false negative/positive), lớp có đúng không, các box cùng loại có nhất quán không. |
| Instance segmentation | **Một mask (polygon) + một lớp cho mỗi vật**, mỗi vật có `instance_id` riêng. | `traffic-015` (car): có các điểm polygon `[372, 205]`, `[331, 201]`, `[375, 236]` cần kiểm tra thứ tự điểm và hình học; chưa đủ bằng chứng để khẳng định polygon tự cắt. `kitchen-005` (bàn): 598 điểm, lẫn bát và bột. `dining-001` (bàn): 1194 điểm. Cùng một xe ở `traffic` bị `yolo11n.pt` gọi là `car` (0.634) nhưng `yolo11n-seg.pt` gọi là `truck` (0.50) ở gần cùng vị trí `[407, 198, 484, 288]`. | Vẽ polygon bám sát đường viền phần nhìn thấy, tách riêng từng vật theo quy tắc visible/amodal của guideline, kiểm tra vùng chồng lấn và vật bị che. | Phủ mask lên ảnh để xem: biên có lọt nền hay cắt vào vật không, polygon có tự cắt không, hai vật chạm nhau đã tách đúng chưa, lớp có đúng không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh mẫu COCO được cấp trong bài lab (có ghi nguồn trong `IMAGE_ATTRIBUTION.md`) và không ghi họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân nào vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng lại, không xử lý, không lưu và không đưa lên repo, rồi báo ngay cho **Lab Coach** (giảng viên hướng dẫn lab).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
